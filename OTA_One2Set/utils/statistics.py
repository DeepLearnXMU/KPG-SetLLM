import math
import numpy as np
import torch

def trg_learning_info(trg_lengths: list, reorder_cols, has_null: list):
    """
        统计目标短语的学习情况（有多少个被学习了，总共有多少个gt）
        gt利用率（覆盖率） = 一组target中参与学习的短语的个数 ÷ 该组target中总的短语个数
        precise：只统计真正的trg（不包括手动append的null）学习情况
        rough：包括null一起统计

        * Params:
        
            `trg_lengths`: 每个batch有多少个target

            `reorder_cols`: OT分配的结果，每个control code对应gt的索引

            `has_null`: 该组target中是否有手动添加null

        * Returns: (`n_precise_trg_learned`, `n_precise_trg`, `n_rough_trg_learned`, `n_rough_trg`)

            `n_precise_trg_learned`: 不包括null，当前batch被学习的gt个数

            `n_precise_trg`: 不包括null，当前batch总的gt个数

            `n_rough_trg_learned`: 包括null，当前batch被学习的gt个数

            `n_rough_trg`: 包括null，当前batch总的gt个数
    """
    precise, rough = [], []
    for b, lenth in enumerate(trg_lengths):
        trg_learned = torch.unique(reorder_cols[b])
        rough.append(trg_learned.shape[0])

        if has_null[b]:  # 如果 null 被学习了的话，则在 precise 指标下不统计 null
            trg_learned = torch.masked_select(trg_learned, trg_learned != lenth - 1)  
        precise.append(trg_learned.shape[0])
        
    return sum(precise), sum(trg_lengths) - sum(has_null), sum(rough), sum(trg_lengths)


def cc_learning_info(trg_lengths: list, reorder_cols, has_null: list):
    """
        统计control_code的学习情况（有多少个control code学了null，多少个学了null，本轮有多少个control code）

        * Params:

            `trg_lengths`: 每个batch有多少个target，如果has_null为true，则最后一个target为null

            `reorder_cols`: OT分配的结果，每个control code对应gt的索引

            `has_null`: 该组target中是否有手动添加null

        * Returns: (`n_cc_null`, `n_cc_gt`)

            `n_cc_null`: 有多少control code被分配给了null

            `total_cc_num`: B个batch共多少个control code
    """

    null_indices = reorder_cols.new_tensor(trg_lengths).view(-1, 1) - 1
    has_null = reorder_cols.new_tensor(has_null).view(-1, 1)
    null_assignment = (reorder_cols == null_indices) * has_null
    n_cc_null = null_assignment.sum()
    total_cc_num = reorder_cols.size(0) * reorder_cols.size(1)
    
    return n_cc_null.item(), total_cc_num


def learning_info(trg_lengths: list, reorder_cols, has_null: list):
    return trg_learning_info(trg_lengths, reorder_cols, has_null) + cc_learning_info(trg_lengths, reorder_cols, has_null)


class LossStatistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, 
                 loss=0.0, 
                 n_tokens=0, 
                 n_batch=0, 
                 forward_time=0.0, 
                 loss_compute_time=0.0, 
                 backward_time=0.0, 
                 pre_assignment_info=(0., 0., 0., 0., 0., 0.),  # (n_precise_trg_learned, n_precise_trg, n_rough_trg_learned, n_rough_trg, n_cc_assignment_null, n_total_control_code)
                 ab_assignment_info=(0., 0., 0., 0., 0., 0.),
                 supply_gt_demand_cnt=0):
        self.loss = loss
        if math.isnan(loss):
            raise ValueError("Loss is NaN")
        self.n_tokens = n_tokens
        self.n_batch = n_batch
        self.forward_time = forward_time
        self.loss_compute_time = loss_compute_time
        self.backward_time = backward_time
        self.pre_assignment_info = np.array(pre_assignment_info)
        self.ab_assignment_info = np.array(ab_assignment_info)
        self.supply_gt_demand_cnt = supply_gt_demand_cnt

    def update(self, stat):
        """
        Update statistics by suming values with another `LossStatistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        if math.isnan(stat.loss):
            raise ValueError("Loss is NaN")
        self.n_tokens += stat.n_tokens
        self.n_batch += stat.n_batch
        self.forward_time += stat.forward_time
        self.loss_compute_time += stat.loss_compute_time
        self.backward_time += stat.backward_time
        self.pre_assignment_info += stat.pre_assignment_info
        self.ab_assignment_info += stat.ab_assignment_info
        self.supply_gt_demand_cnt += stat.supply_gt_demand_cnt
        
    def xent(self):
        """ compute normalized cross entropy """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return self.loss / self.n_tokens

    def ppl(self):
        """ compute normalized perplexity """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return math.exp(min(self.loss / self.n_tokens, 100))

    def total_time(self):
        return self.forward_time, self.loss_compute_time, self.backward_time

    def pre_trg_coverage(self):
        return self.pre_assignment_info[0] / self.pre_assignment_info[1], self.pre_assignment_info[2] / self.pre_assignment_info[3]
    
    def ab_trg_coverage(self):
        return self.ab_assignment_info[0] / self.ab_assignment_info[1], self.ab_assignment_info[2] / self.ab_assignment_info[3]

    def cc_null_assignment_ratio(self):
        return (self.pre_assignment_info[4] + self.ab_assignment_info[4]) / (self.pre_assignment_info[5] + self.ab_assignment_info[5])

    def clear(self):
        self.loss = 0.0
        self.n_tokens = 0
        self.n_batch = 0
        self.forward_time = 0.0
        self.loss_compute_time = 0.0
        self.backward_time = 0.0
        self.pre_assignment_info = np.zeros((6,))
        self.ab_assignment_info = np.zeros((6,))
        self.supply_gt_demand_cnt = 0


if __name__ == "__main__":
    trg_lengths = [5, 4]
    reorder_cols = torch.tensor([[1, 3, 1, 4, 2, 0, 3, 0, 3, 3],
                                 [1, 3, 1, 0, 3, 3, 3, 1, 3, 0]])
    
    print(trg_learning_info(trg_lengths, reorder_cols, [0, 1]))  # (7, 8, 8, 9)

