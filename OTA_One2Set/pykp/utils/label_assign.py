import torch
import numpy as np
from functools import partial
from multiprocessing.pool import ThreadPool
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment

EPS = 1e-8


def hungarian_assign(decode_dist, target, ignore_indices, random=False):
    """
    :param decode_dist: (batch_size, max_kp_num, kp_len, vocab_size)
    :param target: (batch_size, max_kp_num, kp_len)
    :return:
    """
    batch_size, max_kp_num, kp_len = target.size()
    reorder_rows = torch.arange(batch_size)[..., None]
    if random:
        reorder_cols = np.concatenate([np.random.permutation(max_kp_num).reshape(1, -1) for _ in range(batch_size)], axis=0)
    else:
        score_mask = target.new_zeros(target.size()).bool()
        for i in ignore_indices:
            score_mask |= (target == i)
        score_mask = score_mask.unsqueeze(1)  # (batch_size, 1, max_kp_num, kp_len)

        score = decode_dist.new_zeros(batch_size, max_kp_num, max_kp_num, kp_len)
        
        for b in range(batch_size):
            for l in range(kp_len):
                score[b, :, :, l] = decode_dist[b, :, l, target[b, :, l]]
        score = score.masked_fill(score_mask, 0)
        score = score.sum(-1)  # batch_size, max_kp_num, max_kp_num

        reorder_cols = []
        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(score[b].detach().cpu().numpy(), maximize=True)
            reorder_cols.append(col_ind.reshape(1, -1))
            # total_score += sum(score[b][row_ind, col_ind])
        reorder_cols = np.concatenate(reorder_cols, axis=0)
    return reorder_rows, reorder_cols


def report_instance_assign_stats(b, trg_kp_num, rematch_cols, has_null, solution, k, score, score_nor, lr_adp=None):
    from utils.statistics import learning_info

    n_precise_trg_learned, n_precise_trg, n_rough_trg_learned, n_rough_trg, n_cc_assignment_null, n_total_control_code \
        = learning_info([trg_kp_num], rematch_cols[-1], [has_null[b]])
    
    coe_var = (
        torch.std(score_nor, dim=0) / torch.mean(score_nor, dim=0) 
        if trg_kp_num != 1 
        else 0
    )

    precise_coverage = (n_precise_trg_learned + EPS) / (n_precise_trg + EPS)
    rough_coverage = (n_rough_trg_learned + EPS) / (n_rough_trg + EPS)
    
    cc_null_assigment_ratio = n_cc_assignment_null / n_total_control_code

    report = f'''
        [{b}]: 
        rematch_cols: {rematch_cols[-1]}
        trg_kp_num: {trg_kp_num}
        coverage:{(precise_coverage, rough_coverage)}
        k: {k}
        score:{score}
        score_nor: {score_nor}
        coefficient of variation: {coe_var}
        solution: {solution}
        control code null assignment ratio: {cc_null_assigment_ratio}
    '''
    print(report)
    # print(f'[{b}]: \nrematch_cols: {rematch_cols[-1]}\ntrg_kp_num: {trg_kp_num}\ncoverage:{(precise_coverage, rough_coverage)}\nk: {k}\nscore:{score}\nscore_nor: {score_nor}\ncoefficient of variation: {coe_var}\nsolution: {solution}\nlr_adp: {lr_adp}')


def np_topk(arr, k, axis=0):
    """
    Perform topK based on np.argpartition. 
    :param arr: to be sorted
    :param k: select and sort the top `k` items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    # get top k items in origin order
    idx = np.argpartition(-arr, kth=k, axis=axis)
    idx = idx.take(indices=range(k), axis=axis)
    val = np.take_along_axis(arr, indices=idx, axis=axis)
    # sort the top k items
    sorted_idx = np.argsort(-val, axis=axis)
    idx = np.take_along_axis(idx, indices=sorted_idx, axis=axis)
    val = np.take_along_axis(val, indices=sorted_idx, axis=axis)
    return val, idx


def normalize_score_np(b, score, has_null, temperature, null_weaken_ratio=1.0):
    trg_kp_num, pred_kp_num = score.shape

    score_nor = score.copy()
    if has_null[b]:
        score_nor[-1] *= null_weaken_ratio

    if null_weaken_ratio != 0 or trg_kp_num != 1:  ## 如果null的分数没有被置0，或不止一个gt，此时就不会出现除0的情况
        score_nor = score_nor ** (1 / temperature)
        score_nor = score_nor / np.sum(score_nor, axis=0)# * trg_kp_num * pred_kp_num

    return score_nor


def dynamic_k_strategy_np(b, score_nor, k_strategy: str, has_null, top_candidates, supply_gt_demand_cnt):
    if k_strategy not in ['normal']:
        raise NotImplementedError
    trg_kp_num, pred_kp_num = score_nor.shape

    k = np.ones_like(score_nor, shape=(trg_kp_num, ))  # k[:-1]是ground-truth期望分配得到的control code数, k[-1]是null token对应的control code数
    # 如果不进入下方的分支, 意味着：gt太多了, 以至于每个control code人手学一个gt都还有剩的, 所以这个时候还去学null干啥
    if has_null[b]:  # 其实等价于 trg_kp_num <= pred_kp_num, 意味着 null 需要被学习
        topk_score_nor, _ = np_topk(score_nor, top_candidates, axis=1)
        sum_topk_score_nor = np.sum(topk_score_nor, axis=1)
        k = np.ceil(sum_topk_score_nor)
        left_k = int(pred_kp_num - np.sum(k[:-1]))
        k[-1] = left_k
        # print(f"[{b}]:\n\tk:{k}\n\tleft_k:{left_k}\n\ttopk_score_nor:{topk_score_nor}\n\tscore_nor:{score_nor}")
        if left_k < 0:   # ground-truth期望分配得到的control code数超出了总的control code数
            supply_gt_demand_cnt[b] = -left_k
            # print(-left_k)
            k[-1] = 0.0  # null token不应再分配有k值
            k_sorted_arg = (k - sum_topk_score_nor)[:-1].argsort()[::-1]
            loop_idx, xxcnt = 0, 0
            while left_k < 0:  # 从最高的供应量依次-1, 直到供给不超过现有的总control code数为止
                xxcnt += 1
                if xxcnt > 2000:  # debug
                    print("Error: endless loop in k assignment!")
                    print("[%d]: k_sorted_arg: " % b, k_sorted_arg)
                    print("[%d]: k: " % b, k)
                    print("[%d]: sum(k): " % b, torch.sum(k))
                    print("[%d]: sum_topk_score_nor: " % b, sum_topk_score_nor)
                    print("[%d]: sum(sum_topk_score_nor): " % b, np.sum(sum_topk_score_nor))
                    exit()
                if k[k_sorted_arg[loop_idx]] >= 2:
                    k[k_sorted_arg[loop_idx]] -= 1
                    left_k += 1
                loop_idx = (loop_idx + 1) % (k.shape[0] - 1)
    # debug, check `k`
    if k[:-1].__contains__(0.):
        print("expected `k` are assigned as zero! \n[%d]: k: " % b, k)
        
    return k


def sinkhorn_iter_np(suppliers, demanders, cost, epsilon, n_iterations):
    def M(C, u, v, epsilon):
        return (-C + np.expand_dims(u, -1) + np.expand_dims(v, -2)) / epsilon
    
    u = np.ones_like(suppliers)
    v = np.ones_like(demanders)

    # Sinkhorn iterations
    for _ in range(n_iterations):
        v += epsilon * (np.log(demanders + 1e-8) - logsumexp(M(cost, u, v, epsilon).T, axis=-1))
        u += epsilon * (np.log(suppliers + 1e-8) - logsumexp(M(cost, u, v, epsilon), axis=-1))

    U, V = u, v
    # Transport plan pi = diag(a)*K*diag(b)
    solution = np.exp(M(cost, U, V, epsilon))
    # Sinkhorn distance
    total_cost = np.sum(solution * cost)
    return solution, total_cost


def assign_for_one_instance_np(b, target, decode_dist, pred_kp_num, kp_len, has_null, k_strategy, supply_gt_demand_cnt, epsilon, n_iterations, opt):

    trg = target[b]
    trg_kp_num = trg.shape[0]
    assert trg_kp_num > 0
    
    trg = trg[:, :opt.assign_steps]
    # decode_dist[b, :, l] 即第b个batch输出的任意kp的第l个词（记为 pr_w ）的分布
    # 因此 decode_dist[b, :, l, target[:, l]] 就是上述分布中 pr_w 对应位置的target的概率值
    score = np.zeros((pred_kp_num, trg_kp_num, kp_len))
    for l in range(kp_len):
        score[:, :, l] = decode_dist[b, :, l, trg[:, l]].T
    
    # score_first_step = score[:, :, 0].T
    score = score.sum(-1).T  # trg_kp_num, pred_kp_num

    score_nor = normalize_score_np(
        b, score, 
        has_null, 
        opt.assign_temperature
    )

    k = dynamic_k_strategy_np(
        b, score_nor, 
        k_strategy, has_null,
        opt.top_candidates, 
        supply_gt_demand_cnt
    )

    cost = -score_nor
    if opt.null_cost_zero and has_null[b]:
        cost = np.concatenate([score_nor[:-1], np.zeros_like((1, pred_kp_num))])
    if opt.interrupt_cost:
        interrupt = np.random.randn(*cost.shape) / 10
        cost += interrupt
    solution, _ = sinkhorn_iter_np(k, np.ones_like(k, shape=(pred_kp_num, )), cost, epsilon, n_iterations)

    rematch_col = np.argmax(solution, axis=0).reshape(1, -1)

    if opt.stats_only:
        pass
        # report_instance_assign_stats(b, trg_kp_num, rematch_cols, has_null, solution, k, score, score_nor)

    return rematch_col


def optimal_transport_assign(opt, decode_dist, target, epsilon=1e-3, n_iterations=100, has_null=None):
    """
    # params
        * `decode_dist`: 
            * (batch_size, pred_kp_num, kp_len, vocab_size)
            * each word is a probability distribution with a sample size of `vocab_size`
        * `target`:
            * (batch_size, trg_kp_num, kp_len)
            * each word is an index of vocabulary, ranging from 0 to `vocab_size` - 1
        * `epsilon`:
            * param for Sinkhorn Iteration (the smaller `epsilon`, the more precise matching result is; * however too `epsilon` small would generate `nan`)
        * `n_iterations`:
            * param for Sinkhorn Iteration (number of iterations)
        * `has_null`: 
            * list, `len(has_null) == batch_size`
            * which group of data is appended with background (null token)
    # return
        * `rematch_rows`:
            * a simple list of `[0, 1, ..., batch_size - 1]`.
        * `rematch_cols`:
            * the i-th pr matches the `rematch_cols[i]`-th gt
    """
    
    k_strategy = opt.k_strategy
    assert k_strategy in ["null_protection", "normal"], \
        "k_strategy can only be `null_protection` or `normal`, while k is %s!" % k_strategy

    batch_size, pred_kp_num, kp_len, _ = decode_dist.shape
    supply_gt_demand_cnt = decode_dist.new_zeros((batch_size, ))  # 当前batch中每条数据是否满足“预计总供应量大于总需求量, 需要从最高供应量开始向下-1”

    decode_dist_np = decode_dist.detach().cpu().numpy()
    target_np = [trg.detach().cpu().numpy() for trg in target]
    supply_gt_demand_cnt_np = supply_gt_demand_cnt.detach().cpu().numpy()

    ### multi threads, executing assignment for each instance in a batch in parallel
    ### for accelerating
    pool = ThreadPool(batch_size)
    assign_fn = partial(
        assign_for_one_instance_np, 
        target=target_np, decode_dist=decode_dist_np, 
        pred_kp_num=pred_kp_num, kp_len=kp_len, 
        has_null=has_null, k_strategy=k_strategy, 
        supply_gt_demand_cnt=supply_gt_demand_cnt_np, 
        epsilon=epsilon, n_iterations=n_iterations, 
        opt=opt
    )
    # assign_fn = partial(
    #     assign_for_one_instance, 
    #     target=target, decode_dist=decode_dist, 
    #     pred_kp_num=pred_kp_num, kp_len=kp_len, 
    #     has_null=has_null, k_strategy=k_strategy, 
    #     supply_gt_demand_cnt=supply_gt_demand_cnt, 
    #     epsilon=epsilon, n_iterations=n_iterations, 
    #     opt=opt
    # )
    rematch_cols = pool.map(assign_fn, list(range(batch_size)))
    pool.close()
    pool.join()

    rematch_rows = torch.arange(batch_size)[..., None]
    # rematch_cols = torch.cat(rematch_cols, dim=0)

    rematch_cols = torch.from_numpy(np.concatenate(rematch_cols)).to(opt.device)
    supply_gt_demand_cnt = torch.from_numpy(supply_gt_demand_cnt_np).to(opt.device)

    return rematch_rows, rematch_cols, supply_gt_demand_cnt


if __name__ == '__main__':
    np.set_printoptions(linewidth=250)

    # 测试分数
    score = torch.tensor([[2.9346e-03, 7.3927e-04, 3.7396e-04, 2.1913e-03, 1.9249e-05, 1.2607e-02, 2.8530e-02, 1.0523e-05, 3.4267e-04, 7.1781e-04],
                          [1.0343e-03, 7.6850e-03, 2.4964e-03, 6.4758e-02, 9.4605e-03, 1.6504e-02, 3.0325e-02, 7.2908e-04, 1.9064e-04, 3.8416e-04],
                          [8.6856e-01, 5.1528e-01, 1.6740e-01, 2.5912e-01, 7.5238e-02, 2.9478e-02, 1.7406e-02, 8.4788e-02, 8.2401e-01, 2.3255e-01]]).to("cuda:0")
    # 边界值：trg_kp_num = 1
    # score = torch.tensor([[0.9560, 0.9754, 0.6472, 0.3134, 0.5542, 0.3425, 0.1477, 0.7107, 0.7453, 0.8872]])
    # 边界值：trg_kp_num = 2
    # score = torch.tensor([[3.3779e-04, 4.6938e-03, 2.3928e-03, 1.9540e-03, 7.7704e-01, 1.0710e-02, 3.1798e-03, 7.0305e-02, 1.5052e+00, 5.1333e-04],
    #                       [8.8250e-01, 3.8205e-01, 4.6821e-01, 4.1528e-01, 5.7684e-01, 1.8410e-01, 2.1384e-01, 4.4098e-02, 2.4696e-01, 1.7385e-01]])

    score = score.detach().cpu().numpy()

    trg_kp_num, pred_kp_num = score.shape

    score_nor = normalize_score_np(0, score, has_null=[1], temperature=10, null_weaken_ratio=0.5)

    k = dynamic_k_strategy_np(0, score_nor, "normal", has_null=[1], top_candidates=3, supply_gt_demand_cnt=[0])

    suppliers = k
    demanders = np.ones_like(k, shape=(10, ))

    cnt = 0
    # for _ in range(1000):
    cost = -np.concatenate([score_nor[:-1], np.zeros_like(score, shape=(1, pred_kp_num))])
    # interrupt = np.random.randn(*cost.shape) / 10
    # cost += interrupt
    
    res, _ = sinkhorn_iter_np(suppliers, demanders, cost, 1e-3, 100)

    rematch_col = np.argmax(res, axis=0).reshape(1, -1)

    if (
        np.array(
            [
                (rematch_col == i).sum() 
                for i in range(score.shape[0])
            ], dtype=rematch_col.dtype
        ) != k
    ).any(): 
        cnt += 1
    print(cnt)

    print(score_nor.dtype)
    print(f'score_nor: {score_nor}')
    print(f'cost: {cost}')
    print(f'期望分配情况: {k}')
    print(f'solution: {res}')
    print(f'rematch_col: {rematch_col}')
    print(f'实际分配情况：{np.array([(rematch_col == i).sum() for i in range(score.shape[0])], dtype=rematch_col.dtype)}')

    # pred_dist = torch.gather(score_nor, dim=0, index=rematch_col).squeeze(0)
    # pred_dist_mask = (rematch_col != (len(suppliers) - 1))
    # if len(suppliers) == 1 or not True:  ## 如果trg中只有一个null，则不做mask；如果trg中没有null，也不做mask
    #     pred_dist_mask = torch.ones_like(rematch_col)
    # relative_dist = torch.min(pred_dist / score_nor[-1], torch.ones_like(pred_dist))
    # lr_adp = (relative_dist * pred_dist_mask).sum() / pred_dist_mask.sum()

    # coe_var = torch.std(score_nor, dim=0) / torch.mean(score_nor, dim=0)

    # print(pred_dist)
    # print(pred_dist_mask)
    # print(relative_dist)
    # print(lr_adp)
    # print(coe_var)


