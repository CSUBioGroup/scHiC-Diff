from abc import ABC, abstractmethod
import numpy as np
import anndata as ad
import scanpy as sc
import os.path as osp
import torch
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from scdiff.data.base import MaskDataset
from scdiff.utils.data import mask_data_offline


class DenoisingBase(ABC):
    # === 修改点 1: __init__ 方法重构 ===
    # 增加了 adata=None 参数
    def __init__(self, datadir='./data', seed=10, normalize=True, dataset='Lee', fname='Lee_Neuronal_Cell_1Mb_chr8',
                 splits={'valid': 0.1, 'test': 0.1}, mask_type='mar',
                 force_split=False, post_cond_flag=False, return_raw=False, mask_strategy='none_zero',
                 adata=None):

        self.celltype_key = 'cell_type'
        self.batch_key = 'batch'
        self.datadir = datadir
        self.normalize = normalize
        self.return_raw = return_raw
        self.post_cond_flag = post_cond_flag

        # 核心逻辑：如果外部没有传入 adata 对象，才从硬盘读取
        if adata is None:
            self._read(datadir=datadir, normalize=normalize, dataset=dataset, fname=fname)
        # 如果外部传入了 adata 对象，就直接使用它
        else:
            # 使用 copy() 以免在处理验证集时影响到训练集的数据
            self.adata = adata.copy()
            print("Received pre-loaded anndata object. Processing it now...")
            
            # === 新增：检测是否已预处理 ===
            if self._is_preprocessed(self.adata):
                print("Data is already preprocessed, skipping normalization and log transformation...")
                self._process_preprocessed_adata()
            else:
                print("Data is not preprocessed, applying normalization and log transformation...")
                # 调用新的、只负责处理数据的函数
                self._process_adata(normalize=normalize)

        # 后续步骤对于两种情况都是一样的
        self._prepare_split(splits=splits, seed=seed, fname=fname, mask_strategy=mask_strategy, mask_type=mask_type,
                            force_split=force_split)
        self._init_conditions()
        self._prepare()

    def library_norm(self):
        orig_x = self.adata.X.todense()
        depth_every_cell = np.array(np.sum(orig_x, axis=1)).flatten()
        zero_depth_cells = np.where(depth_every_cell == 0)[0]
        if len(zero_depth_cells) > 0:
            print("存在测序深度为0的细胞！已移除")
            print(zero_depth_cells)
            removed_cell_ids = self.adata.obs_names[zero_depth_cells]
            print("被移除的细胞的行ID为：", removed_cell_ids)
            sc.pp.filter_cells(self.adata, min_genes=1)
            print("剩余细胞数量：", self.adata.shape[0])
        target_depth = float(np.median(depth_every_cell))
        print("target library depth：", target_depth)
        self.adata.X = self.adata.X.astype(np.float64)
        sc.pp.normalize_per_cell(self.adata, counts_per_cell_after=target_depth)

    # def scHiC_normalize(self):
    #     raw_x = torch.tensor(self.adata.X.toarray())
    #     x = torch.log1p(raw_x)
    #     sparse_x = csr_matrix(x)
    #     self.adata.X = sparse_x
    def scHiC_normalize(self):
        # 核心优化：直接对稀疏矩阵的非零数据部分(X.data)进行log1p操作。
        # 这样可以完全避免创建巨大的稠密矩阵，内存效率极高。
        print("Performing log1p transformation directly on sparse data to save memory...")
        
        # 创建一个副本以进行安全修改
        processed_X = self.adata.X.copy()
        
        # 只对非零元素应用 log1p 变换
        processed_X.data = np.log1p(processed_X.data)
        
        # 将处理后的稀疏矩阵赋值回去
        self.adata.X = processed_X

    def _is_preprocessed(self, adata):
        """检测数据是否已经预处理"""
        return ('preprocessing' in adata.uns and 
                adata.uns['preprocessing'].get('preprocessed', False))
    
    def _process_preprocessed_adata(self):
        """处理已预处理的数据"""
        print("Processing already preprocessed data...")
        
        # 确保必要的元数据存在
        if self.celltype_key not in self.adata.obs.columns:
            self.adata.obs[self.celltype_key] = 'unknown'
        if self.batch_key not in self.adata.obs.columns:
            self.adata.obs[self.batch_key] = 0
            
        # 检查必要的layers是否存在
        required_layers = ['counts', 'train_mask', 'valid_mask', 'test_mask']
        for layer in required_layers:
            if layer not in self.adata.layers:
                raise ValueError(f"Preprocessed data missing required layer: {layer}")
        
        print("输入数据中非0元素总数：", np.count_nonzero(self.adata.layers['counts'].A))
        print("预处理数据验证完成")

    # === 修改点 2: 新增一个只负责处理数据的函数 ===
    def _process_adata(self, normalize=True):
        """
        这个新函数包含了所有对已经加载到 self.adata 的 AnnData 对象进行的操作。
        """
        self.adata.var_names_make_unique()
        self.adata.obs[self.celltype_key] = 'unknown'
        self.adata.obs[self.batch_key] = 0
        self.adata.layers['counts'] = self.adata.X.copy()
        print("输入数据中非0元素总数：", np.count_nonzero(self.adata.layers['counts'].A))
        if normalize:
            print("Normalizing by cell library sizes...")
            self.library_norm()
            print("Ln transforming...")
            self.scHiC_normalize()

    # === 修改点 3: _read 函数现在只负责读取和调用处理函数 ===
    def _read(self, datadir='./data', normalize=True, dataset='Lee', fname='Lee_Neuronal_Cell_1Mb_chr8'):
        if osp.exists(osp.join(datadir, fname)) and fname.endswith('.h5ad'):
            self.adata = ad.read_h5ad(osp.join(datadir, fname))
        else:
            raise ValueError("输入文件不存在！")
        
        # 读取后，调用新的处理函数
        self._process_adata(normalize=normalize)

    def _prepare_split(self, splits, mask_strategy='random', mask_type='mar',
                       seed=10, fname='Denoising_processed.h5ad', force_split=False):
        # 检查是否已经有预处理的掩码
        if self._is_preprocessed(self.adata):
            print("Using preprocessed masks from data...")
            # 验证掩码是否存在
            required_masks = ['train_mask', 'valid_mask', 'test_mask']
            for mask_name in required_masks:
                if mask_name not in self.adata.layers:
                    raise ValueError(f"Preprocessed data missing required mask: {mask_name}")
            
            # 打印掩码统计信息
            train_mask = self.adata.layers['train_mask']
            valid_mask = self.adata.layers['valid_mask']
            test_mask = self.adata.layers['test_mask']
            
        else:
            # 对于非预处理数据，需要 splits 参数
            assert 'train' in splits and 'valid' in splits, "splits must contain 'train' and 'valid' keys"
            
            print("Generating new masks...")
            self.adata.obs['split'] = 'train'
            print("Data mask strategy：", mask_strategy, ' & ',  mask_type)
            train_mask, valid_mask = mask_data_offline(self.adata, mask_strategy, mask_type,
                                                                  valid_mask_rate=splits['valid'], seed=seed)
            test_mask = np.ones(self.adata.shape, dtype=bool)
            self.adata.layers['train_mask'] = train_mask
            self.adata.layers['valid_mask'] = valid_mask
            self.adata.layers['test_mask'] = test_mask
            
        print("train set: ", np.sum(train_mask))
        print("valid set: ", np.sum(valid_mask))
        print("train set + valid set ", np.sum(train_mask) + np.sum(valid_mask))
        print("test set: ", np.sum(test_mask))

    def _init_conditions(self):
        self.celltype_enc = LabelEncoder()
        self.celltype_enc.classes_ = np.array(sorted(self.adata.obs[self.celltype_key].astype(str).unique()))
        self.batch_enc = LabelEncoder()
        self.batch_enc.classes_ = np.array(sorted(self.adata.obs[self.batch_key].astype(str).unique()))
        if self.post_cond_flag:
            self.cond_num_dict = {'cell_type': len(self.celltype_enc.classes_)}
            self.post_cond_num_dict = {'batch': len(self.batch_enc.classes_)}
        else:
            self.cond_num_dict = {'batch': len(self.batch_enc.classes_),'cell_type': len(self.celltype_enc.classes_)}
            self.post_cond_num_dict = None

    def _load(self):
        self.input = torch.tensor(self.adata.X.A if self.normalize else self.adata.layers['counts'].A).float()
        if self.SPLIT == 'test':
            self.target = self.input.clone()
        mask = self.adata.layers[f'{self.SPLIT}_mask']
        # 处理稀疏矩阵的情况
        if hasattr(mask, 'toarray'):
            mask = mask.toarray()
        self.mask = mask
        if self.SPLIT != 'test':
            train_mask = self.adata.layers['train_mask']
            # 处理稀疏矩阵的情况
            if hasattr(train_mask, 'toarray'):
                train_mask = train_mask.toarray()
            train_mask = train_mask.astype(bool)
            # 转换为 torch tensor 以确保索引操作正确
            train_mask_tensor = torch.from_numpy(train_mask)
            self.input[~train_mask_tensor] = 0.
        if self.normalize and self.return_raw:
            self.raw_input = self.adata.layers['counts'].A
        self.celltype = self.celltype_enc.transform(self.adata.obs[self.celltype_key].astype(str))
        self.batch = self.batch_enc.transform(self.adata.obs[self.batch_key].astype(str))
        self.cond = {'batch': torch.tensor(self.batch).float(),'cell_type': torch.tensor(self.celltype).float(),}

    @abstractmethod
    def _prepare(self):
        ...


class DenoisingTrain(MaskDataset, DenoisingBase):
    SPLIT = "train"
    
    def __init__(self, *args, **kwargs):
        # 显式调用 DenoisingBase 的初始化
        DenoisingBase.__init__(self, *args, **kwargs)

class DenoisingValidation(MaskDataset, DenoisingBase):
    SPLIT = "valid"
    
    def __init__(self, *args, **kwargs):
        # 显式调用 DenoisingBase 的初始化
        DenoisingBase.__init__(self, *args, **kwargs)

class DenoisingTest(MaskDataset, DenoisingBase):
    SPLIT = "test"
    
    def __init__(self, *args, **kwargs):
        # 显式调用 DenoisingBase 的初始化
        DenoisingBase.__init__(self, *args, **kwargs)