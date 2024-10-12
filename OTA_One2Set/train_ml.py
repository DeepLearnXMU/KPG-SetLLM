import os
import sys
import time
import math
import logging

import torch
import torch.nn as nn

import pykp.utils.io as io
from inference.evaluate import evaluate_loss
from pykp.utils.label_assign import hungarian_assign, optimal_transport_assign
from pykp.utils.masked_loss import masked_cross_entropy
from utils.functions import time_since
from utils.report import export_train_and_valid_loss
from utils.statistics import LossStatistics, learning_info

EPS = 1e-8


def train_model(model, optimizer, train_data_loader, valid_data_loader, opt):
    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    total_train_loss_statistics = LossStatistics()
    report_train_loss_statistics = LossStatistics()
    report_train_ppl = []
    report_valid_ppl = []
    report_train_loss = []
    report_valid_loss = []
    best_valid_ppl = float('inf')
    best_valid_loss = float('inf')
    num_stop_dropping = 0
    
    if opt.stats_only:
        model.eval()
    else:
        model.train()

    training_range = range(1) if opt.stats_only else range(opt.start_epoch, opt.epochs + 1)
    for epoch in training_range:
        if early_stop_flag:
            break

        logging.info(f"len of train_data_loader: {len(train_data_loader)}")
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1

            batch_loss_stat = train_one_batch(batch, model, optimizer, opt)
            report_train_loss_statistics.update(batch_loss_stat)
            total_train_loss_statistics.update(batch_loss_stat)

            if total_batch % (opt.report_every / 10) == 0:
                current_train_ppl = report_train_loss_statistics.ppl()
                current_train_loss = report_train_loss_statistics.xent()
                logging.info(
                        "Epoch %d; batch: %d; total batch: %d, avg training ppl: %.3f, loss: %.3f" % (epoch, batch_i,
                                                                                                      total_batch,
                                                                                                      current_train_ppl,
                                                                                                      current_train_loss))
                
                supply_gt_demand_cnt = report_train_loss_statistics.supply_gt_demand_cnt
                total_supply_gt_demand_cnt = total_train_loss_statistics.supply_gt_demand_cnt
                logging.info(
                        "batch: %d; supply_gt_demand_cnt: %d; total batch: %d; total supply_gt_demand_cnt: %d" % (batch_i, 
                                                                                                                  supply_gt_demand_cnt, 
                                                                                                                  total_batch, 
                                                                                                                  total_supply_gt_demand_cnt))
                
                if total_batch > 1:
                    logging.info(
                            "present: precise trg coverage: %.3f, rough trg coverage: %.3f" % report_train_loss_statistics.pre_trg_coverage())
                    logging.info(
                            "absent: precise trg coverage: %.3f, rough trg coverage: %.3f" % report_train_loss_statistics.ab_trg_coverage())
                    logging.info(
                            "control code null assignment ratio: %.3f" % report_train_loss_statistics.cc_null_assignment_ratio())
                     
            if not opt.stats_only and epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and
                         total_batch % opt.checkpoint_interval == 0):
                    valid_loss_stat = evaluate_loss(valid_data_loader, model, opt)
                    model.train()

                    current_valid_loss = valid_loss_stat.xent()
                    current_valid_ppl = valid_loss_stat.ppl()
                    logging.info("Enter check point!")

                    current_train_ppl = report_train_loss_statistics.ppl()
                    current_train_loss = report_train_loss_statistics.xent()

                    # debug
                    if math.isnan(current_valid_loss) or math.isnan(current_train_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (
                                epoch, batch_i, total_batch))
                        exit()

                    if current_valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
                        logging.info("Valid loss drops")
                        sys.stdout.flush()
                        best_valid_loss = current_valid_loss
                        best_valid_ppl = current_valid_ppl
                        num_stop_dropping = 0

                        check_pt_model_path = os.path.join(opt.model_path, 'best_model.pt')
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)
                    else:
                        num_stop_dropping += 1
                        logging.info("Valid loss does not drop, patience: %d/%d" % (
                            num_stop_dropping, opt.early_stop_tolerance))

                        # decay the learning rate by a factor
                        for i, param_group in enumerate(optimizer.param_groups):
                            old_lr = float(param_group['lr'])
                            new_lr = old_lr * opt.learning_rate_decay
                            if old_lr - new_lr > EPS:
                                param_group['lr'] = new_lr

                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        ' * avg training ppl: %.3f; avg validation ppl: %.3f; best validation ppl: %.3f' % (
                            current_train_ppl, current_valid_ppl, best_valid_ppl))
                    logging.info(
                        ' * avg training loss: %.3f; avg validation loss: %.3f; best validation loss: %.3f' % (
                            current_train_loss, current_valid_loss, best_valid_loss))

                    report_train_ppl.append(current_train_ppl)
                    report_valid_ppl.append(current_valid_ppl)
                    report_train_loss.append(current_train_loss)
                    report_valid_loss.append(current_valid_loss)

                    if num_stop_dropping >= opt.early_stop_tolerance:
                        logging.info(
                            'Have not increased for %d check points, early stop training' % num_stop_dropping)
                        early_stop_flag = True
                        break
                    report_train_loss_statistics.clear()

            if opt.stats_only:
                report_train_loss_statistics.clear()
    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_loss(report_train_loss, report_valid_loss, report_train_ppl, report_valid_ppl,
                                opt.checkpoint_interval, train_valid_curve_path)
    
    total_supply_gt_demand_cnt = total_train_loss_statistics.supply_gt_demand_cnt
    total_cc_null_assignment_ratio = total_train_loss_statistics.cc_null_assignment_ratio()
    logging.info(
        "total batch: %d; total supply_gt_demand_cnt: %d; total control code null assignment ratio: %.3f" % (
            total_batch, total_supply_gt_demand_cnt, total_cc_null_assignment_ratio
        )
    )


def train_one_batch(batch, model, optimizer, opt):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
    trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _ = batch
    # print("---------------- DEBUG: trg_mask ----------------")
    # print("trg_mask[0] in dataloader: ", trg_mask[0])
    # print("trg_mask.shape: ", trg_mask.shape)
    # print("trg.device: ", trg.device)
    # print("---------------- DEBUG: trg_mask ----------------")

    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
    batch_size = src.size(0)
    word2idx = opt.vocab['word2idx']
    target = trg_oov if opt.copy_attention else trg

    optimizer.zero_grad()
    start_time = time.time()
    if opt.fix_kp_num_len:
        y_t_init = target.new_ones(batch_size, opt.max_kp_num, 1) * word2idx[io.BOS_WORD]
        if opt.set_loss:  # K-step target assignment
            model.eval()
            with torch.no_grad():
                memory_bank = model.encoder(src, src_lens, src_mask)
                state = model.decoder.init_state(memory_bank, src_mask)
                control_embed = model.decoder.forward_seg(state)

                input_tokens = src.new_zeros(batch_size, opt.max_kp_num, opt.assign_steps + 1)
                decoder_dists = []
                input_tokens[:, :, 0] = word2idx[io.BOS_WORD]
                for t in range(1, opt.assign_steps + 1):
                    decoder_inputs = input_tokens[:, :, :t]
                    decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(opt.vocab_size - 1),
                                                                word2idx[io.UNK_WORD])

                    decoder_dist, _ = model.decoder(decoder_inputs, state, src_oov, max_num_oov, control_embed)
                    input_tokens[:, :, t] = decoder_dist.argmax(-1)
                    decoder_dists.append(decoder_dist.reshape(batch_size, opt.max_kp_num, 1, -1))

                decoder_dists = torch.cat(decoder_dists, -2)

                if opt.seperate_pre_ab:
                    mid_idx = opt.max_kp_num // 2
                    
                    if opt.use_optimal_transport:
                        
                        background = torch.tensor(
                            [word2idx[io.NULL_WORD]] + [word2idx[io.PAD_WORD]] * (opt.max_kp_len - 1)
                        ).to(opt.device)
                        bg_mask = torch.tensor([1] + [0] * (opt.max_kp_len - 1)).to(opt.device)
                        pre_targets, pre_trg_masks, ab_targets, ab_trg_masks = [], [], [], []  # 一个batch里面每组数据对应的target数量是不一样的，用tensor无法对齐，所以用list
                        pre_has_null, ab_has_null = [False] * batch_size, [False] * batch_size
                        for b in range(batch_size):
                            pre_target, pre_trg_mask, ab_target, ab_trg_mask = [], [], [], []

                            for t in range(mid_idx):
                                if any(target[b, t] != background): # 去掉null kp，只保留真实的target kp
                                    pre_target.append(list(target[b, t]))
                                    pre_trg_mask.append(list(trg_mask[b, t]))
                            
                            if len(pre_target) != opt.max_kp_num // 2:
                                pre_has_null[b] = True
                                pre_target.append(list(background))  # 补上一个null kp作为学习的目标
                                pre_trg_mask.append(list(bg_mask))

                            pre_targets.append(torch.tensor(pre_target).to(opt.device))
                            pre_trg_masks.append(torch.tensor(pre_trg_mask).to(opt.device))

                            for t in range(mid_idx, opt.max_kp_num):
                                if any(target[b, t] != background):
                                    ab_target.append(list(target[b, t]))
                                    ab_trg_mask.append(list(trg_mask[b, t]))
                            
                            if len(ab_target) != opt.max_kp_num // 2:
                                ab_has_null[b] = True
                                ab_target.append(list(background))  # 补上一个null kp作为学习的目标
                                ab_trg_mask.append(list(bg_mask))

                            ab_targets.append(torch.tensor(ab_target).to(opt.device))
                            ab_trg_masks.append(torch.tensor(ab_trg_mask).to(opt.device))

                        # present trg assignment
                        _, pre_reorder_cols, pre_supply_gt_demand_cnt = optimal_transport_assign(
                            opt, 
                            decoder_dists[:, :mid_idx], 
                            pre_targets, 
                            has_null=pre_has_null
                        )
                    
                        # absent trg assignment
                        _, ab_reorder_cols, ab_supply_gt_demand_cnt = optimal_transport_assign(
                            opt, 
                            decoder_dists[:, mid_idx:],
                            ab_targets, 
                            has_null=ab_has_null
                        )
                        
                        # 统计 control code 与 target 的匹配情况
                        pre_assignment_info = learning_info(
                            [pre_targets[i].shape[0] for i in range(batch_size)],
                            pre_reorder_cols, 
                            pre_has_null
                        )
                        ab_assignment_info = learning_info(
                            [ab_targets[i].shape[0] for i in range(batch_size)], 
                            ab_reorder_cols, 
                            ab_has_null
                        )

                        new_pre_targets, new_pre_trg_masks, new_ab_targets, new_ab_trg_masks = [], [], [], []
                        for b in range(batch_size):
                            new_pre_targets.append(pre_targets[b][pre_reorder_cols[b]])
                            new_pre_trg_masks.append(pre_trg_masks[b][pre_reorder_cols[b]])
                            new_ab_targets.append(ab_targets[b][ab_reorder_cols[b]])
                            new_ab_trg_masks.append(ab_trg_masks[b][ab_reorder_cols[b]])

                        target[:, :mid_idx] = torch.stack(new_pre_targets, axis=0)
                        trg_mask[:, :mid_idx] = torch.stack(new_pre_trg_masks, axis=0)
                        target[:, mid_idx:] = torch.stack(new_ab_targets, axis=0)
                        trg_mask[:, mid_idx:] = torch.stack(new_ab_trg_masks, axis=0)

                        
                    else:
                        pre_reorder_index = hungarian_assign(decoder_dists[:, :mid_idx],
                                                            target[:, :mid_idx, :opt.assign_steps],
                                                            ignore_indices=[word2idx[io.NULL_WORD],
                                                                            word2idx[io.PAD_WORD]])
                        target[:, :mid_idx] = target[:, :mid_idx][pre_reorder_index]
                        trg_mask[:, :mid_idx] = trg_mask[:, :mid_idx][pre_reorder_index]

                        ab_reorder_index = hungarian_assign(decoder_dists[:, mid_idx:],
                                                            target[:, mid_idx:, :opt.assign_steps],
                                                            ignore_indices=[word2idx[io.NULL_WORD],
                                                                            word2idx[io.PAD_WORD]])
                        target[:, mid_idx:] = target[:, mid_idx:][ab_reorder_index]
                        trg_mask[:, mid_idx:] = trg_mask[:, mid_idx:][ab_reorder_index]

                        # 统计 control code 与 target 的匹配情况
                        null_kp = decoder_dists.new_tensor(
                            [word2idx[io.NULL_WORD]] + [word2idx[io.PAD_WORD]] * (opt.max_kp_len - 1)
                        )
                        n_cc_null = (target == null_kp).all(-1)

                        EPS = 1e-8
                        pre_assignment_info = (EPS, EPS, EPS, EPS, n_cc_null.sum().item(), target.size(0) * target.size(1))
                        ab_assignment_info = (EPS, EPS, EPS, EPS, EPS, EPS)
                        pre_supply_gt_demand_cnt, ab_supply_gt_demand_cnt = torch.tensor([0]), torch.tensor([0])

                        # n_cc = target.size(0) * target.size(1)
                        # print(f'{n_cc_null.shape = }\n{n_cc = }\n{n_cc_null = }')
                        # print(f'cc_null_ratio = {pre_assignment_info[-2] / pre_assignment_info[-1]}')

                else:
                    if opt.use_optimal_transport:
                        reorder_index = optimal_transport_assign(decoder_dists, target[:, :, :opt.assign_steps],
                                                                 [word2idx[io.NULL_WORD],
                                                                  word2idx[io.PAD_WORD]])
                        target = target[reorder_index]
                        trg_mask = trg_mask[reorder_index]
                    else:
                        reorder_index = hungarian_assign(decoder_dists, target[:, :, :opt.assign_steps],
                                                         [word2idx[io.NULL_WORD],
                                                          word2idx[io.PAD_WORD]])
                        target = target[reorder_index]
                        trg_mask = trg_mask[reorder_index]
                
            if opt.stats_only:
                model.eval()
            else:
                model.train()

        memory_bank = model.encoder(src, src_lens, src_mask)
        state = model.decoder.init_state(memory_bank, src_mask)
        control_embed = model.decoder.forward_seg(state)

        input_tgt = torch.cat([y_t_init, target[:, :, :-1]], dim=-1)
        input_tgt = input_tgt.masked_fill(input_tgt.gt(opt.vocab_size - 1), word2idx[io.UNK_WORD])
        decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov, control_embed)
    else:
        y_t_init = trg.new_ones(batch_size, 1) * word2idx[io.BOS_WORD]  # [batch_size, 1]
        input_tgt = torch.cat([y_t_init, trg[:, :-1]], dim=-1)
        memory_bank = model.encoder(src, src_lens, src_mask)
        state = model.decoder.init_state(memory_bank, src_mask)
        decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov)

    if opt.adaptive_lr_scale:
        # select the correctly predicted slots
        tokensum = input_tokens[:, :, 1:].sum(-1).unsqueeze(-1)
        tokensum = tokensum.repeat((1, 1, opt.max_kp_num))

        target_tokensum = target[:, :, :opt.assign_steps].sum(-1).unsqueeze(1)
        target_tokensum = target_tokensum.repeat((1, opt.max_kp_num, 1))
        
        correct_slots = (tokensum == target_tokensum).sum(-1).float()
        keyphrase_mask = (target[:, :, 0] == word2idx[io.NULL_WORD]).float()
        
        keyphrase_tokensum = input_tokens[:, :, 1:].sum(-1).unsqueeze(-1)
        keyphrase_tokensum = keyphrase_tokensum.repeat((1, 1, opt.max_kp_num))
        keyphrase_tokensum = torch.einsum('bnj,bn->bnj', tokensum, 1-keyphrase_mask)
        keyphrase_tokensum = torch.einsum('bjn,bn->bjn', keyphrase_tokensum, 1-keyphrase_mask)
        key_correct_slots = (keyphrase_tokensum == target_tokensum).sum(-1).float()
        
        correct_null_slots = 1 - torch.einsum('bn,bn->bn', correct_slots, keyphrase_mask)
        correct_null_slots = torch.where(correct_null_slots < 0,
                                        correct_null_slots.\
                                        new_zeros(correct_null_slots.shape),
                                        correct_null_slots).detach()
        
        exact_correct_slots = (tokensum == target_tokensum).float()
        exact_correct_slots = torch.einsum('bnl,bn->bnl', exact_correct_slots, keyphrase_mask)
        
        present_correct_slots_num = exact_correct_slots[:, :mid_idx, :mid_idx].sum(-1)
        absent_correct_slots_num = exact_correct_slots[:, mid_idx:, mid_idx:].sum(-1)
        correct_slots_num = torch.cat([present_correct_slots_num, absent_correct_slots_num], dim=-1)

        exact_correct_slots = torch.nonzero(exact_correct_slots)
        
        one_matrix = correct_null_slots.new_ones(correct_null_slots.shape)
        for exact_correct_slot in exact_correct_slots:
            batch_i = exact_correct_slot[0]
            slot_i = exact_correct_slot[1]
            if correct_slots_num[batch_i, slot_i] == 1 and key_correct_slots[batch_i, slot_i] == 0:
                one_matrix[batch_i, slot_i] = 0
        one_matrix = one_matrix.detach()
        correct_null_slots = (correct_null_slots + ((correct_slots_num == 1) & (one_matrix == 0)).float()).detach() 

        # caculate the posibility of token null with new func
        decoder_dist_reshape = decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)
        decoder_dist_ = decoder_dist_reshape[:, :, 0, :]
        decoder_dist_null_probability = decoder_dist_[:, :, word2idx[io.NULL_WORD]]
        decoder_dist_first_token_rate = decoder_dist_null_probability
        
        target_first_token = target[:, :, 0]
        target_first_token_mask = (target_first_token == word2idx[io.NULL_WORD])
        temp = torch.where(target_first_token_mask, decoder_dist_first_token_rate,\
                            decoder_dist_[:, :, word2idx[io.NULL_WORD]]) 
        
        decoder_dist_ = torch.cat([decoder_dist_[:,:,:word2idx[io.NULL_WORD]],\
                                temp.unsqueeze(-1), decoder_dist_[:,:,word2idx[io.NULL_WORD]+1:]], dim=-1).unsqueeze(2)
        
        decoder_dist_reshape = torch.cat([decoder_dist_, decoder_dist_reshape[:, :, 1:, :]], dim=2)

        # scall the loss of token null using the under-estimation of other keyphrase token 
        decoder_dist_predict = decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)
        decoder_dist_predict = torch.gather(decoder_dist_predict, dim=-1, index=target.unsqueeze(-1))
        decoder_dist_predict_ = decoder_dist_predict[:, :, 0, 0]
        
        decoder_dist_predict_under_estimation = (decoder_dist_predict_ + EPS) / (decoder_dist_null_probability + EPS)
        
        mask_normal = (decoder_dist_predict_under_estimation >= 1)
        degree_predict_under_estimation = torch.where(mask_normal,\
                                        decoder_dist_predict_under_estimation.\
                                        new_ones(decoder_dist_predict_under_estimation.shape),\
                                        decoder_dist_predict_under_estimation)
        
        decoder_dist_predict_under_estimation = torch.einsum('bn,bn->bn',\
                                                degree_predict_under_estimation,\
                                                1-(target[:,:,0] == word2idx[io.NULL_WORD]).float()).sum(-1) /\
                                                ((1-(target[:,:,0] == word2idx[io.NULL_WORD]).float()).sum(-1) + EPS)
        
        decoder_dist_predict_under_estimation = torch.where(decoder_dist_predict_under_estimation==0,\
                                        decoder_dist_predict_under_estimation.\
                                        new_ones(decoder_dist_predict_under_estimation.shape),\
                                        decoder_dist_predict_under_estimation)

    forward_time = time_since(start_time)
    start_time = time.time()
    if opt.fix_kp_num_len:
        if opt.seperate_pre_ab:
            mid_idx = opt.max_kp_num // 2

            if not opt.adaptive_lr_scale:
                decoder_dist_reshape = decoder_dist.reshape(batch_size, opt.max_kp_num, opt.max_kp_len, -1)

            pre_loss = masked_cross_entropy(
                decoder_dist_reshape[:, :mid_idx]\
                    .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                target[:, :mid_idx].reshape(batch_size, -1),
                trg_mask[:, :mid_idx].reshape(batch_size, -1),
                loss_scales=[opt.loss_scale_pre],
                scale_indices=[word2idx[io.NULL_WORD]]
            )
            
            ab_loss = masked_cross_entropy(
                decoder_dist_reshape[:, mid_idx:]
                    .reshape(batch_size, opt.max_kp_len * mid_idx, -1),
                target[:, mid_idx:].reshape(batch_size, -1),
                trg_mask[:, mid_idx:].reshape(batch_size, -1),
                loss_scales=[opt.loss_scale_ab],
                scale_indices=[word2idx[io.NULL_WORD]]
            )
            
            if opt.adaptive_lr_scale:
                pre_loss = pre_loss.reshape(batch_size, opt.max_kp_num//2, opt.max_kp_len)
                ab_loss = ab_loss.reshape(batch_size, opt.max_kp_num//2, opt.max_kp_len)
                loss = torch.cat([pre_loss, ab_loss],dim=1)
                loss = torch.einsum('bnl,bn->bnl', loss, correct_null_slots)
                
                loss_scall = decoder_dist_predict_under_estimation.unsqueeze(-1).unsqueeze(-1).detach() * loss
                loss = torch.where(target == word2idx[io.NULL_WORD], loss_scall, loss)
            else:
                loss = pre_loss + ab_loss

            loss = loss.sum()

        else:
            loss = masked_cross_entropy(decoder_dist, target.reshape(batch_size, -1), trg_mask.reshape(batch_size, -1),
                                        loss_scales=[opt.loss_scale], scale_indices=[word2idx[io.NULL_WORD]])
    else:
        loss = masked_cross_entropy(decoder_dist, target, trg_mask)
    loss_compute_time = time_since(start_time)

    total_trg_tokens = trg_mask.sum().item()
    total_trg_sents = src.size(0)
    if opt.loss_normalization == "tokens":  # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches':  # use batch_size to normalize the loss
        normalization = total_trg_sents
    else:
        raise ValueError('The type of loss normalization is invalid.')
    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    total_loss = loss.div(normalization)

    if not opt.stats_only:
        total_loss.backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

    if not opt.stats_only:
        optimizer.step()
        
    supply_gt_demand_cnt = torch.count_nonzero(pre_supply_gt_demand_cnt + ab_supply_gt_demand_cnt)

    stat = LossStatistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time,
                          loss_compute_time=loss_compute_time, backward_time=backward_time, 
                          pre_assignment_info=pre_assignment_info, 
                          ab_assignment_info=ab_assignment_info,
                          supply_gt_demand_cnt=supply_gt_demand_cnt)
    return stat
