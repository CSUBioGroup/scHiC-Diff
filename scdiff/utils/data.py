import warnings
from functools import partial
# 举个例子，如果你有一个接受三个参数的函数，使用 partial，你可以创建一个新的函数，这个新函数预设了其中一个参数的值，因此在调用新函数时只需要提供剩余的参数。
from itertools import chain, product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz



def cal_depth_factor(raw_x):
    raw_x = raw_x.cpu()
    total_counts_every_cell = raw_x.sum(dim=1, keepdim=True)
    if (total_counts_every_cell == 0).any():
        raise ValueError("存在测序深度为0的细胞！")
    else:
        depth_factor = torch.median(total_counts_every_cell) / total_counts_every_cell
        return depth_factor


# 适用于 scHi-C data 的 normalize方法
def scHiC_normalize(x):
    x = x.cpu()
    # x = x * depth_factor
    x = torch.log1p(x)
    return x


def inverse_scHiC_normalize(x):
    x = x.cpu()
    x = torch.expm1(x)   # exp (minus 1)
    # x = x / depth_factor
    return x


def save_result(res_tensor, path, file_name):
    res_tensor = res_tensor.cpu()
    rows, cols = res_tensor.nonzero(as_tuple=True)
    values = res_tensor[rows, cols]
    # 构造CSR矩阵
    csr = csr_matrix((values, (rows, cols)), shape=res_tensor.shape)
    # 保存为NPZ文件
    save_npz(f"{path}/{file_name}.npz", csr)



def get_candidate_conditions(
    context_cond_candidates_cfg: Optional[DictConfig],
    le_dict: Dict[str, LabelEncoder],
) -> torch.Tensor:
    # NOTE: currently only support celltype and batch conditions
    if context_cond_candidates_cfg is None:
        warnings.warn(
            "context_cond_candidates_cfg not specified, using 'grid' mode by default",
            UserWarning,
            stacklevel=2,
        )
        mode, options = "grid", None
    else:
        mode = context_cond_candidates_cfg.mode
        options = context_cond_candidates_cfg.get("options", None)

    if mode == "grid":
        # Example config option
        # mode: grid  # use all possible conditions
        cond = torch.LongTensor(list(product(*[range(le_dict[k].classes_.size) for k in sorted(le_dict)])))

    elif mode == "select":
        # Example config option
        # mode: select  # select specific combinations of conditions
        # options:
        #   - [batch1, celltype1]
        #   - [batch5, celltype2]
        cond_list = [
            le_dict[k].transform(np.array(options.get(k, le_dict[k].classes_)))
            for k in sorted(le_dict)
        ]
        cond = torch.LongTensor(list(map(list, zip(*cond_list))))

    elif mode == "partialgrid":
        # Example config option
        # mode: partialgrid  # use the specified options and grid the rest
        # options:
        #   cond1:
        #     - celltype1
        #     - celltype2
        cond_list = [
            le_dict[k].transform(np.array(options.get(k, le_dict[k].classes_)))
            for k in sorted(le_dict)
        ]
        cond = torch.LongTensor(list(product(*cond_list)))

    else:
        raise ValueError(f"Unknown mode {mode!r}, supported options are: "
                         "['grid', 'select', partialgrid]")

    cond = {
        sorted(le_dict)[i]: cond[:, i] for i in range(len(le_dict))
    }
    return cond




'''
none-zero  MAR 数值越小越容易缺失
返回值：三个mask，是二维的ndarray，形状等于adata.X的形状，需要被mask的元素就是True

strategy: random 以及 none-zero:只对非零元素应用掩码
none-zero: 可以选择 MAR(Missing At Random)  以及 MCAR(Missing Completely At Random)

MAR: 数据的缺失是和观测到的数据本身有关的，但和其他未观测数据没有关系。
采用指数分布（减函数）来决定数据点缺失的概率。较小的观测值更有可能被选为缺失，而较大的观测值缺失的概率较小。

MCAR: 当数据缺失是MCAR时，数据的缺失完全是随机的，和任何观测或未观测的数据都没有关系。
使用均匀分布来决定哪些数据点会被标记为缺失。这意味着每个数据点被随机选中作为缺失数据的概率是相等的。
'''
def mask_data_offline(adata: AnnData, mask_strategy: Optional[str] = "random", mask_type: Optional[str] = "mar",
                      valid_mask_rate: Optional[float] = 0.1, seed: Optional[int] = 10):


    def _get_probs(vec, distr='exp'):
        from scipy.stats import expon  # 导入了指数分布的概率密度函数
        return {
            "exp": expon.pdf(vec, 0, 20),   # x,loc,scale
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(distr)  # distribution是哪个就返回哪个
    '''
    如果 x 是一个单一数字，expon.pdf(x) 将返回一个浮点数，表示在 x 处的概率密度。
    如果 x 是一个数组或序列，expon.pdf(x) 将返回一个 numpy.ndarray，其形状与 x 相同，数组中的每个元素都是输入中对应元素的概率密度值。
    '''
    rng = np.random.default_rng(seed)
    # 1. 对于本身由ndarray构建的anndata
    # feat = adata.layers['counts']
    # 2. 对于由稀疏矩阵构建的anndata，才需要转换为dense
    feat = adata.layers['counts'].A   # .A 属性是用于将稀疏矩阵转换为一个密集的（dense）numpy 数组的快捷方式。

    if mask_strategy == 'none_zero':
        print("Mask 20% of non-zero elements...")
        '''  1. 初始化2种mask   1 for mask '''
        train_mask = np.ones(feat.shape, dtype=bool)  # train_mask一开始全部是True，抽完剩下的就都是Train
        valid_mask = np.zeros(feat.shape, dtype=bool)   # valid_mask一开始全部是Zero，抽完需要设置True

        row, col = np.nonzero(feat)  # 找出数组 feat 中非零元素的位置
        nonzero_elements = np.array(feat[row, col])  # 把非零元素压缩成一维数组了，expon.pdf()函数的输入

        num_nonzeros = len(row)
        n_valid = int(np.floor(num_nonzeros * valid_mask_rate))

        # Randomly mask positive counts according to masking probability.
        if mask_type == "mcar":
            distr = "uniform"
        elif mask_type == "mar":
            distr = "exp"
        else:
            raise NotImplementedError(f"Expect mask_type in ['mar', 'mcar'], but found {mask_type}")

        # 概率归一化
        mask_prob = _get_probs(nonzero_elements, distr)  # 此时所有概率之和相加小于1
        mask_prob = mask_prob / sum(mask_prob)  # Normalize，使所有概率之和相加等于1
        # 因为使用rng.choice()函数要求概率数组之和必须等于1

        ''' 2. 抽取数据 ==> valid_mask  & train_mask '''
        # a原数组索引（从0开始）  # 返回数组的长度（抽取的数量）  # a中每个元素被抽取的概率  # replace:抽取后是否放回（一个元素就可能被抽取多次）
        valid_idx = rng.choice(np.arange(num_nonzeros), n_valid, p=mask_prob, replace=False)
        train_mask[row[valid_idx], col[valid_idx]] = False
        valid_mask[row[valid_idx], col[valid_idx]] = True

    # elif mask_strategy == 'random':
    #     test_mask = rng.random(feat.shape) < (test_mask_rate + valid_mask_rate)
    #     valid_mask = test_mask.copy()
    #
    #     nonzero_idx = np.where(test_mask)
    #     test_to_val_ratio = test_mask_rate / (test_mask_rate + valid_mask_rate)
    #     split_point = int(nonzero_idx[0].size * test_to_val_ratio)
    #     test_idx, val_idx = np.split(rng.permutation(nonzero_idx[0].size), [split_point])
    #     # 随机排列所有非零索引，以确保分配的随机性。
    #     # np.split 根据 split_point 将索引分为两部分：测试索引 (test_idx) 和验证索索引 (val_idx)。
    #
    #     test_mask[nonzero_idx[0][val_idx], nonzero_idx[1][val_idx]] = False
    #     valid_mask[nonzero_idx[0][test_idx], nonzero_idx[1][test_idx]] = False
    #     train_mask = ~(test_mask | valid_mask)

    else:
        raise NotImplementedError(f'Unsupported mask_strategy {mask_strategy}')

    return train_mask, valid_mask


'''
函数接受一个字典作为输入，该字典的值是张量，然后它将这些张量转换成一个单一的张量。
常用于批数据加载

torch.stack 沿着一个新的维度进行堆叠，这个新维度是添加到堆叠张量之前的。
如果列表中的每个张量形状相同，且为 (a, b)，那么堆叠后的张量形状将会是 (n, a, b)，其中 n 是列表中张量的数量。
'''
def dict_of_tensors_to_tensor(input_dict):
    tensor_list = []
    for key in sorted(input_dict):
        tensor_list.append(input_dict[key])
    return torch.stack(tensor_list).T   # 如果超过二维，会转置最后两个维度


def dict_to_list_of_tuples(input_dict):
    if len(list(input_dict)) > 1:
        input_list = [input_dict[k] for k in input_dict.keys()]
        return list(map(tuple, zip(*input_list)))
    else:
        return input_dict[list(input_dict)[0]]
