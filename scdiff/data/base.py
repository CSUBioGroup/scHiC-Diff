###
"""
Base classes for datasets.
"""
###
from abc import abstractmethod
from math import ceil
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import Dataset, IterableDataset

from scdiff.utils.data import get_candidate_conditions, dict_of_tensors_to_tensor


class SplitDataset(Dataset):
    SPLIT: Optional[str] = None

    # === 修改点 ===
    # 在构造函数中增加 adata=None 参数
    def __init__(self, adata=None, *args, **kwargs):
        # 不调用 super().__init__，因为 Dataset 的 __init__ 不需要参数
        pass

    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, index):
        item_dict = {
            "input": self.input[index],
            "cond": {k: self.cond[k][index] for k in list(self.cond)},
        }
        if getattr(self, "normalize", False) and getattr(self, "return_raw", False):
            item_dict['raw_input'] = self.raw_input[index]
        if all(hasattr(self, i) for i in ('G_go', 'G_go_weight')):
            item_dict["aug_graph"] = dict(G_go=self.G_go, G_go_weight=self.G_go_weight)
        if getattr(self, "extras", None) is not None:
            item_dict["extras"] = self.extras
        return item_dict

    def _prepare(self):
        print("self.SPLIT=", self.SPLIT)
        assert self.SPLIT is not None, "Please specify SPLIT class attr."
        
        # 检查是否有 split 列
        if "split" in self.adata.obs.columns:
            print("np.unique(self.adata.obs['split']):", np.unique(self.adata.obs["split"]))
            if self.SPLIT in np.unique(self.adata.obs["split"]):
                self.adata = self.adata[self.adata.obs["split"] == self.SPLIT]
        else:
            # 预处理数据没有 split 列，直接使用 mask 来区分
            print("No 'split' column found - using preprocessed masks directly")
        
        self._load()


class MaskDataset(SplitDataset):
    SPLIT: Optional[str] = None
    
    # === 修改点 ===
    # 同样增加 adata=None 参数以保持继承链的一致性
    def __init__(self, adata=None, *args, **kwargs):
        # 直接调用父类，传递所有参数
        super().__init__(adata=adata, *args, **kwargs)

    def __getitem__(self, index):
        item_dict = {
            "input": self.input[index],
            "cond": {k: self.cond[k][index] for k in list(self.cond)},
            "mask": self.mask[index],
        }
        if self.SPLIT == 'test':
            item_dict['masked_target'] = self.target[index]
        if self.normalize and self.return_raw:
            item_dict['raw_input'] = self.raw_input[index]
        return item_dict


class FullDatasetMixin:
    SPLIT = "train"

    def __init__(self, *args, **kwargs):
        kwargs["splits"] = {"train": 1.0, "valid": 0.0, "test": 0.0}
        super().__init__(*args, **kwargs)


class SCIterableBaseDataset(IterableDataset):
    def __init__(self, num_cells=0, valid_ids=None, size=256):
        super().__init__()
        self.num_cells = num_cells
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size
        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class TargetDataset(SplitDataset):
    SPLIT: Optional[str] = None
    TARGET_KEY = "target"

    # === 修改点 ===
    def __init__(self, adata=None, *args, **kwargs):
        # 直接调用父类，传递所有参数
        super().__init__(adata=adata, *args, **kwargs)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        item_dict = super().__getitem__(index)
        if self.target is not None:
            if len(self.target) == len(self.input):
                item_dict[self.TARGET_KEY] = self.target[index]
            else:
                item_dict[self.TARGET_KEY] = self.target
        if self.SPLIT != 'train' and hasattr(self, 'gene_names'):
            item_dict['gene_names'] = self.gene_names
        return item_dict


class GenerationDataset(IterableDataset):
    """Cell generation task dataset."""
    def __init__(
        self,
        use_split: str = "train",
        context_cond_candidates_cfg: Optional[DictConfig] = None,
        generation_cond_candidates_cfg: Optional[DictConfig] = None,
        batch_size: Optional[int] = 4096,
        dropout: float = 0.0,
        n_trials: int = 1,
        n_batches_to_generate: int = 1,
        **kwargs,
    ):
        self.use_split = use_split
        self.context_cond_candidates_cfg = context_cond_candidates_cfg
        self.generation_cond_candidates_cfg = generation_cond_candidates_cfg
        self.batch_size = batch_size
        self.dropout = dropout
        self.n_trials = n_trials
        self.n_batches_to_generate = n_batches_to_generate
        super().__init__(**kwargs)

    def _prepare(self):
        if self.use_split != "all":
            assert self.use_split in np.unique(self.adata.obs["split"])
            self.adata = self.adata[self.adata.obs["split"] == self.use_split]
        self._load()

        self.context_cond_candidates = get_candidate_conditions(
            self.context_cond_candidates_cfg,
            self.le_dict,
        )
        self.generation_cond_candidates = get_candidate_conditions(
            self.generation_cond_candidates_cfg,
            self.le_dict,
        )

    def __iter__(self):
        """Iterator for preparing context and query pairs."""
        n_batches_to_generate = self.n_batches_to_generate
        context_cond_candidates = dict_of_tensors_to_tensor(self.context_cond_candidates)
        generation_cond_candidates = dict_of_tensors_to_tensor(self.generation_cond_candidates)
        cond_tensor = dict_of_tensors_to_tensor(self.cond)
        context_candidate_ind = (cond_tensor.unsqueeze(0)
                                 == context_cond_candidates.unsqueeze(1)).all(-1).any(0)
        context_candidate_idx = torch.where(context_candidate_ind)[0]
        num_context_cells = len(context_candidate_idx)
        batch_size = self.batch_size or len(context_candidate_idx)
        assert batch_size >= len(generation_cond_candidates)
        cond = generation_cond_candidates.repeat(
            ceil(batch_size / len(generation_cond_candidates)), 1)
        query_cond = cond[:batch_size]
        query_cond = {
            sorted(self.cond)[i]: query_cond[:, i] for i in range(len(self.cond))
        }

        for _ in range(self.n_trials):
            for _ in range(n_batches_to_generate):
                select_idx = torch.randint(len(context_candidate_idx), (batch_size,))
                x = F.dropout(self.input[select_idx], self.dropout)
                cell_ids = self.adata.obs.iloc[select_idx].index.tolist()
                yield {"input": x, "cond": query_cond, "context_cell_ids": cell_ids}


class PerturbationDataset(Dataset):
    SPLIT: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return {
            "input": self.input[index],
            "target": self.target[index],
            "cond": self.cond[index],
            "cond_names": self.cond_names,
            "cond_mapping_dict": self.cond_mapping_dict,
            "top_de_dict": self.top_de_dict
        }

    def _prepare(self):
        assert self.SPLIT is not None, "Please specify SPLIT class attr."
        assert self.SPLIT in np.unique(self.adata.obs["split"])
        self._load(self.SPLIT)