import warnings

import scib
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from matplotlib import pyplot
from typing import Dict, List, Optional
from adjustText import adjust_text
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

from scdiff.typing import TensorArray
from scdiff.utils.misc import as_array, as_tensor
from scdiff.utils.data import cal_depth_factor, scHiC_normalize, inverse_scHiC_normalize


'''  样本平均斯皮尔曼相关性系数  '''
def Spearman_corrcoef(x, y):
    """
    计算两个张量之间的样本平均斯皮尔曼相关性系数。

    参数:
    x (torch.Tensor): 形状为 (n_samples, n_features) 的张量
    y (torch.Tensor): 形状为 (n_samples, n_features) 的张量

    返回:
    float: 样本平均斯皮尔曼相关性系数
    """
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # 计算每个样本的斯皮尔曼相关性系数
    corrs = []
    for i in range(x_np.shape[0]):
        corr, _ = spearmanr(x_np[i], y_np[i])
        corrs.append(corr)
    # 计算所有样本的平均相关性系数
    mean_corr = np.mean(corrs)
    return mean_corr



''' Pearson  样本 ZScore    样本与样本直接按的平均线性相关性  '''
def PearsonCorr(y_pred, y_true):
    y_true_c = y_true - torch.mean(y_true, 1)[:, None]
    y_pred_c = y_pred - torch.mean(y_pred, 1)[:, None]
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c, 1)
        / torch.sqrt(torch.sum(y_true_c * y_true_c, 1))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c, 1))
    )
    return pearson


'''  Pearson  元素与元素之间  '''
def PearsonCorr1d(y_true, y_pred):
    y_true_c = y_true - torch.mean(y_true)
    y_pred_c = y_pred - torch.mean(y_pred)
    pearson = torch.nanmean(
        torch.sum(y_true_c * y_pred_c)
        / torch.sqrt(torch.sum(y_true_c * y_true_c))
        / torch.sqrt(torch.sum(y_pred_c * y_pred_c))
    )
    return pearson


@torch.inference_mode()
def evaluate_annotation(
    true: TensorArray,
    pred: TensorArray,
    name: Optional[str],
) -> Dict[str, float]:
    true_array = as_array(true, assert_type=True)
    pred_array = as_array(pred, assert_type=True)

    le = LabelEncoder()
    le.classes_ = np.array(sorted(set(np.unique(true_array).tolist() + np.unique(pred_array).tolist())))

    true = torch.LongTensor(le.transform(true_array))
    pred = torch.LongTensor(le.transform(pred_array))

    num_classes = le.classes_.size
    # num_classes = int(max(true.max(), pred.max())) + 1
    # num_unique_classes = max(true.unique().numel(), pred.unique().numel())
    # if (num_classes == num_unique_classes + 1) and (0 not in true):
    #     warnings.warn(
    #         "Implicitly removing null label (index 0)",
    #         UserWarning,
    #         stacklevel=2,
    #     )
    #     true, pred, num_classes = true - 1, pred - 1, num_classes - 1
    # elif num_classes != num_unique_classes:
    #     warnings.warn(
    #         f"Number of unique classes {num_unique_classes} mismatch the "
    #         f"number of classes inferred by max index {num_classes}",
    #         UserWarning,
    #         stacklevel=2,
    #     )

    suffix = "" if name is None else f"_{name}"

    out = {}
    out[f"acc{suffix}"] = multiclass_accuracy(true, pred, num_classes).item()
    out[f"f1{suffix}"] = multiclass_f1_score(true, pred, num_classes).item()
    out[f"precision{suffix}"] = multiclass_precision(true, pred, num_classes).item()
    out[f"recall{suffix}"] = multiclass_recall(true, pred, num_classes).item()

    return out


# 计算全部元素的RMSE
def all_rmse(pred, true):
    return F.mse_loss(pred, true).sqrt()

# 只计算Mask元素的RMSE，适用于Inpaint任务
def masked_rmse(pred, true, mask):
    pred_masked = pred * mask
    true_masked = true * mask
    size = mask.sum()
    return (F.mse_loss(pred_masked, true_masked, reduction='sum') / size).sqrt()

'''   Z Score 归一化  使得每一行的数据，都服从标准正态分布 '''
# 这种标准化方式逐行考虑，使每一行的数据点（在掩码指示为有效的情况下）具有零均值和单位方差。
def masked_stdz(x, mask):
    size = mask.sum(1, keepdim=True).clamp(1)
    x = x * mask
    # 元素-样本均值  中心化
    x_ctrd = x - (x.sum(1, keepdim=True) / size) * mask
    # NOTE: multiplied by the factor of sqrt of N
    # （标准差） = 原值-均值）2求和再开根号
    x_std = x_ctrd.pow(2).sum(1, keepdim=True).sqrt()
    return x_ctrd / x_std


def masked_corr(pred, true, mask):
    pred_masked_stdz = masked_stdz(pred, mask)
    true_masked_stdz = masked_stdz(true, mask)
    # 这种方法实际上是计算了标准化后的预测值和真实值之间的协方差，并将其标准化。
    # 由于数据已被标准化至均值为零、标准差为一，所以这里的协方差等价于皮尔逊相关系数。
    corr = (pred_masked_stdz * true_masked_stdz).sum(1).mean()
    return corr


@torch.inference_mode()
def denoising_eval(true: TensorArray, pred: TensorArray, raw_x: TensorArray):
    true = as_tensor(true, assert_type=True)
    pred = as_tensor(pred, assert_type=True)
    raw_x = as_tensor(raw_x, assert_type=True)


    global_corr_normed_all =  PearsonCorr1d(pred, true).item()


    corr_normed_all = PearsonCorr(pred, true).item()
    rmse_normed_all = F.mse_loss(pred, true).sqrt().item()
    Spearman_corr_normed_all = Spearman_corrcoef(pred, true)

    # 相关系数 (r-value)：这是一个介于 -1 和 1 之间的数，描述了两个变量之间的线性相关性。
    # 如果 r 接近 1 或 -1，表示变量之间存在强烈的正线性或负线性关系；如果 r 接近 0，则表示几乎没有线性关系。
    r_all = scipy.stats.linregress(pred.ravel().cpu().numpy(), true.ravel().cpu().numpy())[2]
    r_normed_all = scipy.stats.linregress(pred.cpu().numpy().flatten(), true.cpu().numpy().flatten()).rvalue


    ################################
    # depth_factor = cal_depth_factor(raw_x)
    denoise_recon_inv = inverse_scHiC_normalize(pred)

    # 计算全部元素的
    global_corr_orig_all = PearsonCorr1d(denoise_recon_inv, raw_x).item()
    corr_orig_all = PearsonCorr(denoise_recon_inv, raw_x).item()
    rmse_orig_all = F.mse_loss(denoise_recon_inv, raw_x).sqrt().item()
    Spearman_corr_orig_all = Spearman_corrcoef(denoise_recon_inv, raw_x)

    return {
        'RMSE': rmse_orig_all,
        'Pearson (样本归一)': corr_orig_all,
        'Pearson (global)': global_corr_orig_all,
        'Spearman (样本归一)': Spearman_corr_orig_all,
        'R^2': r_normed_all ** 2,

        'Normed RMSE': rmse_normed_all,
        'Normed Pearson (样本归一)': corr_normed_all,
        'Normed Spearman (样本归一)': Spearman_corr_normed_all,
        'Normed Pearson (global)': global_corr_normed_all,
        'Normed R^2': r_all ** 2,

    }





def de_eval(true, pred, ctrl, name):
    true_delta = true - ctrl
    pred_delta = pred - ctrl
    # 返回一个字典
    return {
        # MAE
        f'mae_{name}': (pred - true).abs().mean().item(),
        f'mae_delta_{name}': (pred_delta - true_delta).abs().mean().item(),
        # MSE
        f'mse_{name}': F.mse_loss(pred, true).item(),
        f'mse_delta_{name}': F.mse_loss(pred_delta, true_delta).item(),
        # RMSE
        f'rmse_{name}': np.sqrt(F.mse_loss(pred, true).item()),
        f'rmse_delta_{name}': np.sqrt(F.mse_loss(pred_delta, true_delta).item()),
        # Correlation
        f'corr_{name}': PearsonCorr1d(pred, true).item(),
        f'corr_delta_{name}': PearsonCorr1d(pred_delta, true_delta).item(),
    }




def dict_of_arrays_to_tensor(input_dict):
    tensor_list = []
    for key in sorted(input_dict):
        tensor_list.append(torch.tensor(input_dict[key]))
    return torch.stack(tensor_list).T
# value都是形状为（cell_num，）的一维Tensor
# 两个拼接后，是形状为（n_conditions，cell_num）的tensor，因为torch.stack会创建一个新的维度


def calculate_batch_r_squared(pred, true, conditions):
    conditions = dict_of_arrays_to_tensor(conditions)   # （n_conditions，cell_num）
    unique_cond = conditions.unique(dim=0)  # Returns the unique elements of the input tensor.
    # 先按指定维度将原张量X划分多个子张量x0,x1,x2，..., xn.  再在这些子张量中剔除重复的子张量
    r_squared_list = []
    for i in range(len(unique_cond)):  # 只有一种： [0,0]
        cond_flag = torch.all((conditions == unique_cond[i]), dim=1)
        x = pred[cond_flag].mean(0).numpy()
        y = true[cond_flag].mean(0).numpy()
        _, _, r_value, _, _ = scipy.stats.linregress(x, y)
        r_squared_list.append(r_value ** 2)
    return r_squared_list


def reduce_score_dict_list(score_dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    assert isinstance(score_dict_list, list)

    score_keys = sorted(score_dict_list[0])
    assert all(sorted(i) == score_keys for i in score_dict_list), "All score dicts must contain same score keys"

    scores = {score_key: np.mean([i[score_key] for i in score_dict_list]) for score_key in score_keys}

    return scores
