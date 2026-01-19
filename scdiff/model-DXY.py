"""
Wild mixture of:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8

Thank you!
"""
import warnings
from contextlib import contextmanager
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from scipy.sparse import csr_matrix, save_npz
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from scdiff.modules.diffusion_model import Decoder, Embedder, Encoder
from scdiff.evaluate import (
    denoising_eval
)
from scdiff.modules.ema import LitEma
from scdiff.modules.layers.attention import BasicTransformerBlock
from scdiff.modules.layers.basic import FeedForward
from scdiff.modules.layers.scmodel import EmbeddingDict
from scdiff.utils.diffusion import MaskedEncoderConditioner, timestep_embedding
from scdiff.utils.diffusion import make_beta_schedule
from scdiff.utils.misc import as_1d_vec, exists, count_params, instantiate_from_config
from scdiff.utils.misc import default
from scdiff.utils.modules import create_activation, create_norm
from scdiff.utils.modules import extract_into_tensor, init_weights, mean_flat, noise_like
from scdiff.utils.data import cal_depth_factor, scHiC_normalize, inverse_scHiC_normalize, save_result

RESCALE_FACTOR = np.log(1e4)



class DiffusionModel(nn.Module):
    def __init__(self, save_path, pretrained_gene_list,  input_gene_list=None, dropout=0.,
                 encoder_type='stackffn', embed_dim=512, depth=6, dim_head=64, num_heads=4,
                 decoder_embed_dim=512, decoder_embed_type='linear', decoder_num_heads=4,
                 decoder_dim_head=64, cond_dim=None, cond_tokens=10, cond_type='crossattn', cond_strategy='full_mix',
                 cond_emb_type=None, cond_num_dict=None, cond_mask_ratio=0.5, cond_cat_input=False,
                 post_cond_num_dict=None, post_cond_layers=1, post_cond_norm='layernorm',
                 post_cond_mask_ratio=0.0, norm_layer='layernorm', mlp_time_embed=False, no_time_embed=False,
                 activation='gelu', mask_strategy='random', mask_none_zero=0.67, zero_to_none_zero=0.1,
                 mask_dec_cond=False,mask_dec_cond_ratio=False, mask_dec_cond_se=False, mask_dec_cond_semlp=False,
                 mask_dec_cond_concat=False, mask_value=0, pad_value=0, decoder_mask=None, text_emb=None,
                 text_emb_file=None, freeze_text_emb=True, text_proj_type='linear', text_proj_act=None,
                 stackfnn_glu_flag=False, text_proj_hidden_dim=512, text_proj_num_layers=2, text_proj_norm=None,
                 cond_emb_norm=None, num_perts=None, gears_flag=False, gears_hidden_size=64,
                 gears_mode="single", gears_mlp_layers=2, gears_norm=None, num_go_gnn_layers=1):
        super().__init__()
        self.depth = depth
        self.save_path = save_path

        # --------------------------------------------------------------------------
        # MAE masking options
        self.mask_none_zero = mask_none_zero
        self.zero_to_none_zero =  zero_to_none_zero
        self.mask_strategy = mask_strategy  # random
        self.mask_value = mask_value  # 0
        self.pad_value = pad_value  # 0
        self.decoder_mask = decoder_mask # None


        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        activation = create_activation(activation)  # gelu
        # eg.human chr8 then in_dim = 10731
        self.in_dim = len(pretrained_gene_list) if pretrained_gene_list is not None else len(input_gene_list)
        self.pretrained_gene_list = pretrained_gene_list
        self.input_gene_list = input_gene_list  # None
        # dict's key : BinPair Name, dict's value: index of 0, 1, ....
        pretrained_gene_index = dict(zip(self.pretrained_gene_list, list(range(len(self.pretrained_gene_list)))))  # [0,1,...10730]

        # None
        self.input_gene_idx = torch.tensor([
            pretrained_gene_index[o] for o in self.input_gene_list
            if o in pretrained_gene_index
        ]).long() if self.input_gene_list is not None else None

        assert embed_dim == decoder_embed_dim  # XXX: this seems to be required for MAE (see forward dec)?
        full_embed_dim = embed_dim * cond_tokens   # cond_tokens = 1  embed_dim = 512
        self.post_encoder_layer = Rearrange('b (n d) -> b n d', n=cond_tokens, d=embed_dim)  # b (1 512) -> (b 1 512)

        '''  1. Embedder  '''
        self.embedder = Embedder(pretrained_gene_list, full_embed_dim, 'layernorm', dropout=dropout)

        '''  2. Encoder  (有两个Encoder)   '''
        ####################  Encoder 1   MLP  ####################
        self.encoder_type = encoder_type  # mlp
        if encoder_type == 'attn':
            self.blocks = nn.ModuleList([
                BasicTransformerBlock(full_embed_dim, num_heads, dim_head, self_attn=True, cross_attn=False,
                                      dropout=dropout, qkv_bias=True, final_act=activation)
                for _ in range(depth)])

        # Encoder1 所以是这种情况，定义多层感知机
        elif encoder_type in ('mlp', 'mlpparallel'):
            self.blocks = nn.ModuleList([   # nn.ModuleList 用于容纳这些层，而每一层都是通过 nn.Sequential 来构建的
                nn.Sequential(
                    nn.Linear(full_embed_dim, full_embed_dim),  # 输入维度512, 输出维度512，这表明每个全连接层 输入特征和输出特征数量相等。
                    activation,  #  每个全连接层后都跟随一个激活函数，这有助于引入非线性，使得网络可以学习更复杂的函数。
                    create_norm(norm_layer, full_embed_dim),  # nn.LayerNorm(512)
                ) for _ in range(depth)])  # depth=6

        elif encoder_type in ('stackffn', 'ffnparallel'):
            self.blocks = nn.ModuleList([
                # FeedForward(full_embed_dim, mult=4, glu=False, dropout=dropout)
                nn.Sequential(
                    FeedForward(full_embed_dim, mult=4, glu=False, dropout=dropout),
                    create_norm(norm_layer, full_embed_dim),
                ) for _ in range(depth)])
        elif encoder_type == 'none':
            self.blocks = None
        else:
            raise ValueError(f'Unknown encoder type {encoder_type}')
        # self.encoder_proj = nn.Linear(full_embed_dim, latent_dim)
        # self.norm = create_norm(norm_layer, full_embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim   # 512

        self.time_embed = nn.Sequential(
            nn.Linear(decoder_embed_dim, 4 * decoder_embed_dim),   # 512 -> 2048
            nn.SiLU(),  # Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
            nn.Linear(4 * decoder_embed_dim, decoder_embed_dim),  # 2048 -> 512
        ) if mlp_time_embed else nn.Identity()  # mlp_time_embed = False

        self.no_time_embed = no_time_embed    # False  有Time Embed

        self.cond_type = cond_type
        assert cond_strategy in ("full_mix", "pre_mix")
        self.cond_strategy = cond_strategy
        self.cond_emb_type = cond_emb_type
        self.cond_tokens = cond_tokens
        self.cond_cat_input = cond_cat_input


        if cond_dim is not None or cond_num_dict is not None:
            if cond_emb_type == 'linear':
                assert cond_dim is not None
                self.cond_embed = nn.Sequential(
                    nn.Linear(cond_dim, decoder_embed_dim * cond_tokens),   # (cond_dim, 512)
                    Rearrange('b (n d) -> b n d', n=cond_tokens, d=decoder_embed_dim),
                )
            elif cond_emb_type == 'embedding':   # 细胞注释用的是这个模式
                assert cond_num_dict is not None
                self.cond_embed = EmbeddingDict(cond_num_dict, decoder_embed_dim, depth,
                                                cond_tokens, mask_ratio=cond_mask_ratio,
                                                text_emb=text_emb, text_emb_file=text_emb_file,
                                                norm_layer=cond_emb_norm,
                                                freeze_text_emb=freeze_text_emb,
                                                text_proj_type=text_proj_type,
                                                text_proj_num_layers=text_proj_num_layers,
                                                stackfnn_glu_flag=stackfnn_glu_flag,
                                                text_proj_hidden_dim=text_proj_hidden_dim,
                                                text_proj_act=text_proj_act,
                                                text_proj_norm=text_proj_norm,
                                                # text_proj_dropout=dropout, G_go=G_go,
                                                # G_go_weight=G_go_weight, num_perts=num_perts,
                                                text_proj_dropout=dropout, gears_flag=gears_flag, num_perts=num_perts,
                                                gears_hidden_size=gears_hidden_size, gears_mode=gears_mode,
                                                gears_mlp_layers=gears_mlp_layers, gears_norm=gears_norm,
                                                num_go_gnn_layers=num_go_gnn_layers)
            elif cond_emb_type == 'none':
                self.cond_embed = None
            else:
                raise ValueError(f"Unknwon condition embedder type {cond_emb_type}")
        else:
            self.cond_embed = None

        ####################  Encoder 2  多头交叉注意力机制   ####################
        # depth=6, dim=512, num_heads=8, dim_head=64,
        self.encoder = Encoder(depth, decoder_embed_dim, decoder_num_heads, decoder_dim_head,
                               dropout=dropout, cond_type=cond_type, cond_cat_input=cond_cat_input)

        '''  3. Decoder  '''
        # self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim))
        self.decoder_embed_type = decoder_embed_type  # embedder
        assert decoder_embed_type in ['linear', 'embedder', 'encoder']
        if decoder_embed_type == 'linear':
            self.decoder_embed = nn.Linear(self.in_dim, decoder_embed_dim)

        #     decoder_embed_type: embedder   是这个 √
        elif decoder_embed_type == 'embedder':
            self.decoder_embed = Embedder(pretrained_gene_list, decoder_embed_dim, 'layernorm', dropout=dropout)

        elif decoder_embed_type == 'encoder':
            self.decoder_embed = self.embedder

        self.mask_decoder_conditioner = MaskedEncoderConditioner(
            decoder_embed_dim, mult=4, use_ratio=mask_dec_cond_ratio, use_se=mask_dec_cond_se,
            use_semlp=mask_dec_cond_semlp, concat=mask_dec_cond_concat, disable=not mask_dec_cond)  # disalble = not False = True

        self.decoder_norm = create_norm(norm_layer, decoder_embed_dim)  # LayerNorm  512
        # BatchNorm就是在每个维度上统计所有样本的值，计算均值和方差；LayerNorm就是在每个样本上统计所有维度的值，计算均值和方差

        #         post_cond_layers: 1
        #         post_cond_norm: batchnorm   (not used when post_cond_layers=1)
        #         post_cond_mask_ratio: 0.1
        self.decoder = Decoder(decoder_embed_dim, self.in_dim, dropout, post_cond_norm,
                               post_cond_layers, post_cond_num_dict, act=activation,
                               cond_emb_dim=decoder_embed_dim, cond_mask_ratio=post_cond_mask_ratio)
        # def __init__(self, dim, out_dim, dropout=0., norm_type="layernorm", num_layers=1, cond_num_dict=None,
        #              cond_emb_dim=512, cond_mask_ratio=0., act="gelu", out_act=None):
        # --------------------------------------------------------------------------

        self.initialize_weights()

    # 一次性地对整个模型的参数进行初始化，而不需要 手动为每个层单独设置初始化。
    # 通过继承 torch.nn.Module 类创建的类会自动追踪其属性，只要属性是 torch.nn.Module 类的实例，它就会被认为是子模块。 eg. class Encoder(nn.Module)
    def initialize_weights(self):
        # initialize linear and normalization layers
        self.apply(init_weights)

    # 计算mask中, mask掉了x中的多少0元素，以及非0元素
    def cal_mask_0_number(self, x, mask):
        masked_0 = mask[(x == 0) & (mask == 1)]
        masked_no_0 = mask[(x != 0) & (mask == 1)]


    '''  mask: 0 keep, 1 drop '''
    ''' 大约25%的细胞全部特征被完全mask， 然后再随机mask大约25%的元素 '''
    # TODO: move to DDPM and get mask from there (masking is indepdent on forward)?
    def random_masking(self, x):

        mask = torch.zeros_like(x)
        # 1. 计算应该mask几个0元素和非0元素
        row_ind, col_ind = torch.nonzero(x, as_tuple=True)
        sum_none_zero = len(row_ind)
        n_mask_none_zero = int(sum_none_zero * self.mask_none_zero)   # 1/3
        n_mask_zero = int(n_mask_none_zero * self.zero_to_none_zero)  # 0.1
        # 1. 选中非0的Mask
        selected_indices1 = torch.randperm(len(row_ind))[:n_mask_none_zero]
        mask[row_ind[selected_indices1], col_ind[selected_indices1]] = 1

        # 2. 选中0的mask
        zero_indices = torch.where(x==0)
        selected_indices2 = torch.randperm(len(zero_indices[0]))[:n_mask_zero]
        mask[zero_indices[0][selected_indices2], zero_indices[1][selected_indices2]] = 1
        mask = mask.bool()   # float 转 bool
        x_masked = x * ~mask  # bcz 0 for save, 1 for drop


        return x_masked, mask

    ''' 在 forward 函数中被调用  '''
    def forward_encoder(self, x_ctxt, input_gene_list=None, input_gene_idx=None):
        # 1. embed ctxt
        input_gene_list = default(input_gene_list, self.input_gene_list)
        input_gene_idx = default(input_gene_idx, self.input_gene_idx)
        x_ctxt, gene_idx = self.embedder(x_ctxt, input_gene_list, input_gene_idx)

        # 2. encoder embeded result
        if self.blocks is None:
            hist = [None] * self.depth
        elif self.encoder_type in ("mlpparallel", "ffnparallel"):
            hist = [self.post_encoder_layer(blk(x_ctxt)) for blk in self.blocks]
        # 是这种情况： 调用多层感知机进行encode
        else:
            hist = []
            # 6个全连接层  6个blk因为depth=6
            for blk in self.blocks:  # apply context encoder blocks
                x = blk(x_ctxt)   # Linear(512 to 512) --> gelu --> LayerNorm
                # 每一层得到的还是一个 cell_num x 512 的矩阵
                hist.append(self.post_encoder_layer(x))
                # 给每个block都加个后置层，其实只是Rearange()了一下 变成三维的张量, cell_numx512  --> cell_num x 1 x 512
        return hist, gene_idx




    ''' 在 get_latent 和 forward 函数中被调用
    '''
    def forward_decoder(self, x, context_list, timesteps=None, conditions=None,
                        input_gene_list=None, input_gene_idx=None, aug_graph=None,
                        return_latent=False, mask=None):

        # No. self.decoder_embed_type = embed tokens
        if self.decoder_embed_type == 'linear':
            x = self.decoder_embed(x)

        #  1. 是这种情况： decoder_embed_type: embedder  使用的是我们自己的Embedder类
        else:
            input_gene_list = default(input_gene_list, self.input_gene_list)
            input_gene_idx = default(input_gene_idx, self.input_gene_idx)
            # 调用 Decoder进行编码，还是变成 cell_num x 512的
            x, _ = self.decoder_embed(x, input_gene_list, input_gene_idx)

        # apply masked conditioner
        # 目前设置的 conditioner是 Disable  什么都不改变，还是输入原来的x
        x = self.mask_decoder_conditioner(x, mask)

        # calculate time embedding
        ''' 给每个样本（细胞）添加一个512维度位置信息，方便区分他们，不然神经网络都是一个个无差别处理vector'''
        if timesteps is not None and not self.no_time_embed:  # 满足
            # print("timesteps:", timesteps) 测试的时候len(timesteps)=1，训练和验证时都是长度为样本数的随机打乱的时间步
            timesteps = timesteps.repeat(x.shape[0]) if len(timesteps) == 1 else timesteps
            time_embed = self.time_embed(timestep_embedding(timesteps, self.decoder_embed_dim))  # cell_num x 512
            x = x + time_embed  # 给每个样本数据加上当前时间步特征
            '''  比如如果是文本信息，句子意思是和单词的位置有关的，我们的数据具有时间特征，数据的噪声程度和时间步大小有关  '''
            # x = torch.cat([x, time_embed], dim=0)

        # calculate cell condition embedding (cell type , batch)
        cond_emb_list = None if self.cond_embed is None else self.cond_embed(conditions, aug_graph=aug_graph)
        # cond_emb_list = None
        if not isinstance(cond_emb_list, list):
            cond_emb_list = [cond_emb_list] * self.depth


        # cond_emb_list = [None] * 6 = [None, None, None, None, None, None]
        # 经过6个 BasicTransformerBlock（也就是6次多头交叉注意力机制），进行特征融合
        x = self.encoder(x, context_list, cond_emb_list)

        x_latent = self.decoder_norm(x)  # apply post conditioner layers

        '''  self.decoder: 最后一步，调用Decoder还原原来的形状  '''
        # return x if return_latent else self.decoder(x, conditions)
        return x_latent, self.decoder(x_latent, conditions)   # 我直接两个都返回

    # 求被mask的元素预测的均方误差
    def forward_loss(self, target, pred, mask=None):
        if mask is None:
            mask = torch.ones(target.shape, device=target.device)
        loss = (pred - target) ** 2  # 每个元素的预测误差
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed tokens   # 仅仅求Mask的元素的均方误差
        return loss


    def get_latent(self, x_orig, x, timesteps=None, conditions=None, input_gene_list=None,
                   text_embeddings=None, aug_graph=None, mask=None):
        # embed input
        context_list, _ = self.forward_encoder(x_orig, input_gene_list)
        latent = self.forward_decoder(x, context_list, timesteps, conditions, input_gene_list,
                                      text_embeddings, aug_graph=aug_graph, return_latent=True, mask=mask)
        return latent


    '''
        训练时：
            x_orig是上下文信息（random mask掉的62.5%），x是t时刻的噪声图， mask:62.5%对应
            p_losses中：
                _, model_out, _ = self.model(x_inp_ctxt, x_inp_noised, t=t, conditions=conditions, mask=mask)
        测试时：
             _, denoise_recon, _ = self.model(x, x_noised, t_sample, conditions,
                              input_gene_list, text_embeddings,
                              aug_graph=aug_graph, mask=False)
            =======================================================================================                    
            x_orig时输入的x（充当上下文信息）
            recon: mask="all";    denoise_recon: mask=denoise_mask(tensor)
            x_t是对应于时间步t的噪音图
            denoise_recon = self.sample(x, denoise_t_sample, conditions, input_gene_list,
                                        text_embeddings, aug_graph=aug_graph,
                                        mask=denoise_mask).cpu()
        
    '''
    def forward(self, x_orig, x_t, timesteps=None, conditions=None, input_gene_list=None,
                input_gene_idx=None, aug_graph=None, mask=True):
        # print("== Cian Test ==: forward() 中 x_orig中的非零元素总数：", torch.count_nonzero(x_orig))
        # masking: length -> length * mask_ratio

        # 是Tensor时，时测试步骤 sample()时
        if isinstance(mask, torch.Tensor):   # mask： 0 save  1 drop
            x_ctxt = x_orig * ~mask.bool()  # 不管是训练还是测试：都只是再确保一遍x_ctxt是上下文信息
        elif isinstance(mask, bool):  # 测试阶段，mask = False
            if mask:
                x_ctxt, mask = self.random_masking(x_orig)  # return x_masked, mask
                if self.decoder_mask is not None:  # decoder_mask: inv_enc （ inverse encoder 逆编码器 )
                    if self.decoder_mask == 'enc':
                        x_t[mask.bool()] = self.mask_value

                    elif self.decoder_mask == 'inv_enc':
                        x_t[~mask.bool()] = self.mask_value   # 反向Mask: 那些没被 mask的元素，值设为0

                    elif self.decoder_mask == 'dec':
                        _, dec_mask, _, _ = self.random_masking(x_t)
                        x_t[dec_mask.bool()] = self.mask_value
                        mask = (mask.bool() | dec_mask.bool()).float()
                    else:
                        raise NotImplementedError(f"Unsupported decoder mask choice: {self.decoder_mask}")
            else: # 测试阶段，mask = False
                x_ctxt = x_orig
                mask = torch.zeros_like(x_orig, dtype=bool)  # 0 save 所以 全部save
        elif isinstance(mask, str):
            if mask == "all":
                x_ctxt = x_orig * 0  # XXX: assumes mask value is 0
                mask = torch.ones_like(x_orig, dtype=bool)
            elif mask == "showcontext":
                x_ctxt = x_orig
                mask = torch.ones_like(x_orig, dtype=bool)
            else:
                raise ValueError(f"Unknwon mask type {mask!r}")
        else:
            raise TypeError(f"Unknwon mask specification type {type(mask)}")


        ''' 1. 数据：x_ctxt， 网络：forward_encoder 多层感知机'''
        # 多层感知机，6个全连接层得到的结果存放在context_list
        context_list, gene_idx = self.forward_encoder(x_ctxt, input_gene_list, input_gene_idx)

        ''' 2. 数据：x_t & context_list  网络：forward_decoder 交叉注意力机制  得到预测结果  '''
        latent, pred = self.forward_decoder(x_t, context_list, timesteps, conditions, input_gene_list,
                                    input_gene_idx, aug_graph=aug_graph, mask=mask)


        return latent, pred, mask


'''  因为DiffusionModel是继承的nn.Module类，所以要用pl.LightningModule再包裹一下  '''
class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(self, x_orig, x, timesteps, conditions=None, input_gene_list=None,
                text_embeddings=None, aug_graph=None, mask=True):
        latent, out, mask = self.diffusion_model(x_orig, x, timesteps, conditions, input_gene_list,
                                   text_embeddings, aug_graph, mask)   # 调用扩散模型
        return latent, out, mask

'''  在内部，trainer 会调用 training_step 来处理每个批次的数据，
training_step 调用 forward 方法来获取模型的输出（y_hat），然后根据这个输出和真实标签 y 计算损失。 '''

class ScDiff(pl.LightningModule):
    denoise_target_key: str

    def __init__(self,
                 save_path,
                 model_config,
                 data_normalize=True,
                 return_raw=True,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 loss_strategy="recon_masked",   # recon_full
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 monitor_mode="min",
                 use_ema=True,
                 input_key="input",
                 raw_input_key="raw_input",
                 cond_key="cond",
                 input_gene_list_key="input_gene_list",
                 target_key='target',
                 cond_names_key='cond_names',
                 denoise_mask_key="mask",
                 denoise_target_key="masked_target",
                 text_embeddings_key='text_emb',
                 aug_graph_key='aug_graph',
                 extras_key='extras',
                 cond_names=None,
                 log_every_t=200,
                 in_dim=None,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules  全部都假设有固定的方差？
                 scheduler_config=None,
                 learn_logvar=False,
                 logvar_init=0.,
                 denoise_flag=False,
                 denoise_t_sample=1000,
                 fold_flag=False,
                 test_target_sum=1e3,
                 in_dropout=0.0,
                 cond_to_ignore: list = None,
                 balance_loss=False,
                 path_to_save_fig='./results/hpoly_ddpm_seed10.png',
                 eval_vlb_flag=False,
                 r_squared_flag=True,
                 **kwargs
        ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.save_path = save_path
        self.data_normalize = data_normalize
        self.return_raw = return_raw
        self.denoise_flag = denoise_flag
        self.denoise_t_sample = denoise_t_sample
        self.fold_flag = fold_flag
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.input_key = input_key
        self.cond_key = cond_key
        self.raw_input_key = raw_input_key
        self.input_gene_list_key = input_gene_list_key
        self.target_key = target_key
        self.cond_names_key = cond_names_key
        self.denoise_mask_key = denoise_mask_key
        self.denoise_target_key = denoise_target_key
        self.text_embeddings_key = text_embeddings_key
        self.aug_graph_key = aug_graph_key
        self.extras_key = extras_key
        self.in_dim = in_dim
        self.test_target_sum = test_target_sum
        self.cond_to_ignore = cond_to_ignore
        self.balance_loss = balance_loss
        self.path_to_save_fig = path_to_save_fig
        self.eval_vlb_flag = eval_vlb_flag
        self.r_squared_flag = r_squared_flag
        # 核心还是扩散模型
        self.model = DiffusionWrapper(model_config)  # 就yaml中那些设置
        # 计算参数所占内存
        count_params(self.model, verbose=True)

        self.in_dropout = nn.Dropout(in_dropout)


        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior  # 0
        self.original_elbo_weight = original_elbo_weight  # 0
        self.l_simple_weight = l_simple_weight  # 1

        if monitor is not None:
            self.monitor = monitor
            self.monitor_mode = monitor_mode
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        self.register_buffer("unique_conditions", None)

        self.loss_type = loss_type  # L2 均方误差

        '''  
        重建损失 计算策略
        1. reconstruct masked: 只考虑masked的数据点的重建损失
        2. reconstruct full: 全部数据点
        '''
        assert loss_strategy in ("recon_masked", "recon_full"), f"Unknwon {loss_strategy=}"
        self.loss_strategy = loss_strategy

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        # validation
        self.val_step_outputs = []
        # test
        self.test_step_outputs = []

        self.save_hyperparameters()


    '''  配置和注册模型中使用的一些关键参数和缓冲区  '''
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas  # 这是控制噪声引入水平的关键参数
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)   # [1e-8,4e-4]
        alphas = 1. - betas   # 这是在每个时间步中保留原始数据成分的比例。
        alphas_cumprod = np.cumprod(alphas, axis=0)  # cumulative product
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # 最前面加了一个1而已，再取后面999个
        #  [1.   0.99999999 0.99999998 0.99999996  ....  0.87625944 0.87591172 0.87556345 0.87521462 0.87486523]

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)  # 1000
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        # 创建了新的函数 to_torch，它基于 torch.tensor() 函数，但预设了 dtype=torch.float32 参数。
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # v_posterio=0
        posterior_variance = ((1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) /
                              (1. - alphas_cumprod) + self.v_posterior * betas)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        ''' 就是在这里阐明了两种模式的区别 '''
        # 在 "eps" 参数化中，lvlb_weights 的计算涉及使用模型的 beta 参数（self.betas）、先前计算的 posterior_variance 和 （self.alphas_cumprod）
        # 模型焦点在于精确控制和模拟噪声，这有助于模型在生成数据时更准确地重建原始数据。
        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        # 这种方法直接使用 alphas_cumprod 来估计每个步骤中数据的保留量，并基于此计算权重。
        # 在 "x0" 参数化中，焦点在于如何从带噪声的数据中直接重建出原始数据，而不是模拟整个噪声过程。
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # 参数平滑方案
    # 它在模型训练或验证期间临时切换到使用 EMA 更新的参数。
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())  # 存储了模型当前的参数。这是为了在离开上下文管理器后能够将这些参数恢复到它们原始的状态。
            self.model_ema.copy_to(self.model)  # EMA参数赋给模型
            if context is not None:
                print(f"{context}: Switched to EMA weights")   # 这样方便知道代码进来了
        try:
            # 上下文管理器的核心，它允许代码的执行暂时离开 ema_scope 函数，
            # 去执行 with ema_scope(): 代码块中的内容。在这部分，所有模型的运算都会使用已经替换的 EMA 参数。
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")  # 还原模型的原始参数

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    ''' 根据推断Xt-1的后验公式（关于X0与Xt的）进行均值方差计算'''
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +  # coef1*x0
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t  # coef2*xt
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    ''' p(xt-1,xt)
     根据训练好的扩散模型 来预测原始数据 X0  mask=full
     然后带入，计算得到：p(xt-1,xt)分布的均值和方差'''
    def p_mean_variance(self, x_start, x_t, t, clip_denoised: bool, conditions=None, input_gene_list=None,
                        text_embeddings=None, aug_graph=None):

        if self.cond_to_ignore is not None:  # null
            assert len(self.cond_to_ignore) <= conditions.shape[1]
            assert all([0 <= x_t < conditions.shape[1] for x_t in self.cond_to_ignore])
            conditions[:, self.cond_to_ignore] = 0

        ''' 预测X0(X0 Mode) '''
        ''' self.model 执行 forward 函数'''
        model_latent, model_out, _ = self.model(x_start, x_t, t, conditions=conditions,
                                  input_gene_list=input_gene_list,
                                  aug_graph=aug_graph)

        if self.parameterization == "eps":  # 预测的是噪声，就还得算一下通过这 x0 得到的噪声
            x_recon = self.predict_start_from_noise(x_t, t=t, noise=model_out)
        elif self.parameterization == "x0":  # 预测原始数据X0
            x_recon = model_out
        if clip_denoised:
            # x_recon.clamp_(-1., 1.)
            x_recon.clamp_(0)  # 将所有小于 0 的元素限制为 0，也就是说，最终所有元素都大于等于 0


        # 用q_posterior计算 Xt-1的均值方差，但是x0 = X recon是扩散模型预测的，不是真实的
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    ''' 反向过程（去除噪声）  由X0 Xt 推算Xt-1的分布
        根据Xt和X0，
        t是一个[i]tensor  i属于[500-0]  clip_denoised=True
        返回两个值： 后验分布的数据，变分下界vlb 
    '''

    @torch.no_grad()
    def p_sample(self, x_start, x_t, t, clip_denoised=True, repeat_noise=False, conditions=None,
                 input_gene_list=None, text_embeddings=None, aug_graph=None,
                 calculate_vlb=True):
        b, *_, device = *x_t.shape, x_t.device
        ''' 因为t=0时就是原数据，后面才需要给每个元素均值加一个方差（噪声）'''
        # no noise when t == 0, otherwise nonzero_mask is all one tensor
        nonzero_mask = torch.full((x_t.shape[0], 1), (1 - (t == 0).float()).item()).to(x_t)
        # item()：这个方法将单个元素的张量转换为Python标量。

        # 1. 输入扩散模型，预测X0（denoise_recon),
        # 2. 再计算后验分布： Xt-1 的分布的 mean, var
        model_mean, _, model_log_variance = self.p_mean_variance(x_start=x_start, x_t=x_t, t=t,
                                                                 clip_denoised=clip_denoised,
                                                                 conditions=conditions, aug_graph=aug_graph,
                                                                 input_gene_list=input_gene_list)
        noise = noise_like(x_t.shape, device, repeat_noise)

        if calculate_vlb:  # True
            # 真实输入的X0 ，结合Xt, 算出来的 Xt-1的分布的 q_mean, q_var
            # 作用只是为了计算 vlb
            q_mean, _, q_log_variance = self.q_posterior(x_start, x_t, t)

            # 计算预测分布和真实分布之间的vlb
            vlb = self.normal_kl(q_mean, q_log_variance, model_mean, model_log_variance).cpu()
            vlb = mean_flat(vlb)
            print("Cian Test == : 后验采样时，计算了VLB = ", vlb)
        else:
            vlb = 0

        # pred_xt-1
        '''
            (0.5 * model_log_variance).exp()：将预测的对数方差转换为标准差。
            noise：从标准正态分布中采样的噪声。
            nonzero_mask：确保只有在需要引入噪声的时候（例如 t>1 的情况）才加上噪声，通常在最后一步 t=1 不添加噪声。
        '''
        pred_x_t_minus_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_x_t_minus_1, vlb


    '''  反向去噪的过程， 从1000步的图像开始逐渐去噪   '''
    @torch.no_grad()
    def p_sample_loop(self, x_start, shape, t_start, conditions=None, input_gene_list=None,
                      text_embeddings=None, aug_graph=None, return_intermediates=False,
                      return_vlb=False):

        assert t_start <= self.num_timesteps
        device = self.betas.device
        noise = torch.randn(shape, device=device)
        '''  一开始t_start=1000的时候执行这一步 ，使得x是一个随机噪声  '''
        if t_start == self.num_timesteps:
            # x = noise  # NOTE: this is incorrect for sampling w diff num of query and ctxt cells
            if isinstance(conditions, torch.Tensor):
                N = conditions.shape[0]
            elif isinstance(conditions, dict):  # 是这种dict，既有batch 又有cell_type
                N = conditions[list(conditions)[0]].shape[0]
            x = torch.randn(N, x_start.shape[1], device=device)
            t_start -= 1
        else:  # 利用x input 得到加了t次噪声的图 x
            x = self.q_sample(x_start=x_start, t=t_start, noise=noise)

        intermediates = [x]
        # torch.full((b,), i, device=device, dtype=torch.long)
        vlb_list = []
        # tqdm显示进度条： i = [t_start,0]
        ''' 逆循环： 通常在扩散模型中，这种逆序执行是由模型的反向过程（从随机噪声恢复到原始数据的过程）驱动的。  '''
        for i in tqdm(reversed(range(0, t_start + 1)), total=int(t_start + 1)):
            # 使用训练好的扩散模型预测出X0，以及xt，计算出 Xt-1，赋值给x，
            x, vlb = self.p_sample(x_start, x, torch.tensor([i], device=device, dtype=torch.long),
                                   clip_denoised=self.clip_denoised, conditions=conditions,
                                   input_gene_list=input_gene_list,
                                   text_embeddings=text_embeddings, aug_graph=aug_graph, calculate_vlb=return_vlb)

            vlb_list.append(vlb)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(x)

        if return_vlb:  # True
            return x, torch.stack(vlb_list, dim=1)
        if return_intermediates:
            return x, intermediates
        return x

    # sample() 只用在 test步骤中，训练中不使用，因为是反向过程，从 denoise_t_sample = 1000开始
    @torch.no_grad()
    def sample(self, x_start, t_start, conditions=None, input_gene_list=None,
               text_embeddings=None, aug_graph=None, return_intermediates=False,
               return_vlb=False):
        # mask: 0 for context, 1 for denoise
        in_dim = self.in_dim  # gene总数（bin pair总数）
        shape = (x_start.shape[0], in_dim) if in_dim is not None else x_start.shape
        return self.p_sample_loop(x_start, shape, t_start, conditions=conditions,
                                  input_gene_list=input_gene_list,
                                  text_embeddings=text_embeddings, aug_graph=aug_graph,
                                  return_vlb=return_vlb, return_intermediates=return_intermediates)


    ''' 用正向扩散（添加噪声） 得到t时刻的噪声图 '''
    def q_sample(self, x_start, t, noise=None):
        # print("== Cian Test ==: q_sample()正向扩散，利用X0（有缺失的）计算Xt的图像")
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
               extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        if self.fold_flag:
            out.abs_()
        return out

    # mask是那62.5%的mask，与验证和测试的mask没关系
    def get_loss(self, pred, target, mask=None, mean=True):
        size = mask.numel()  # 得到张量元素总数
        if mask is not None and self.loss_strategy == "recon_masked":
            # print("recon_masked")
            pred = pred * mask
            target = target * mask
            size = mask.sum()

        # print("recon_masked策略下，数据被mask的元素数量是：", size)
        # print("mask的元素的占比：", size/mask.numel())
        # 如果是 recon_full 策略，就不需要改变pred和target
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        if mean:  # False
            loss = loss.sum() / size   # 标量化： 计算整个批次中所有样本的所有特征的总损失。
        return loss



    '''  只在p_losses() 中被调用，即训练时。
         调用时：mask = None
         mask: 0 save  1 mask(drop) '''
    def prepare_noised_input(self, x, t, noise=None, mask=None):
        if noise is None:
            noise = default(noise, lambda: torch.randn_like(x))

        x_inp = self.in_dropout(x)   # 0 drop 数据不变

        ''' mode v2: 完全mask大约25%的细胞全部特征，剩余的75%元素，再随机mask0.5，最终mask掉约0.625的元素 '''
        # 0 for ctxt, 1 for input
        if isinstance(mask, bool) and not mask:
            mask = torch.zeros_like(x)  # use all for context
        # mask=None  训练时进入此  不是Tensor就生成
        elif not isinstance(mask, torch.Tensor):
            _, mask = self.model.diffusion_model.random_masking(x)

        # 1. 提供给扩散模型 的 上下文
        x_inp_ctxt = x_inp * ~mask   # bcz 0 for context

        if isinstance(t, int):
            t = torch.tensor([t], device=x.device)

        # Xt，t有可能是一个一维向量，包含N cell个不同的时刻
        x_inp_noised = self.q_sample(x_start=x_inp, t=t, noise=noise)

        return x_inp_ctxt, x_inp_noised, mask
        # x_inp_ctxt与Mask有关，x_inp_noised与Mask无关


    '''  l_simple_weight = 1  求的完全就是均方误差 '''
    '''  给损失命名的地方也在这里  '''
    def p_losses(self, x_start, t, noise=None, conditions=None, input_gene_list=None,
                 text_embeddings=None, aug_graph=None, target=None):

        noise = default(noise, lambda: torch.randn_like(x_start))

        ''' 1. random_mask
        x_inp_ctxt是mask了 80%+元素的结果，mask与之相对应，x_inp_noised是xt噪声图'''
        x_inp_ctxt, x_inp_noised, ctxt_mask = self.prepare_noised_input(x_start, t, noise)
        # print("== Cian Test ==: 62.5% random mask how many elements?:", ctxt_mask.sum())

        ''' 2. 应用当前参数的扩散模型，得到一个epoch预测结果，预测的是X0 '''
        _, model_out, _ = self.model(x_inp_ctxt, x_inp_noised, timesteps=t, conditions=conditions,
                                  input_gene_list=input_gene_list,
                                  text_embeddings=text_embeddings, aug_graph=aug_graph, mask=ctxt_mask)

        ''' 3. 根据损失函数计算预测结果与真实值之间的误差 '''
        loss_dict = {}
        if target is not None:
            pass
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start   # 我们使用的是这个模式  target=x  是Normalize的
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, ctxt_mask, mean=False)
        # 最终这两个得到的都是[样本数]的tensor
        if not self.balance_loss:
            loss = loss.mean(dim=1)  # dim=1  最终得到的 loss 张量的形状将是 [样本数]，每个元素对应一个样本的平均 MSE 损失。
        else:   # balance_loss=True的时候执行 # 非零值的预测MSE 与 零值的预测MSE 平均一下得到最终的loss值
            print("==Cian Test==: balance loss between loss_nonzero & loss_zero")
            nonzero = target != 0
            loss_nonzero = (loss * nonzero).sum(dim=1) / nonzero.sum(dim=1)   # 某样本中的非零元素的Loss之和 / 样本中特征为非零元素数量
            loss_zero = (loss * ~nonzero).sum(dim=1) / (~nonzero).sum(dim=1)
            loss = (loss_nonzero + loss_zero) / 2
            print("==Cian Test==: balanced loss = ", loss)

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_MSE': loss.mean()})

        # current_elbo_weight = self.vlb_loss_weight_linear_scheduler(self.current_epoch, self.trainer.max_epochs, self.original_elbo_weight, 0.3)
        # self.l_simple_weight = 1-current_elbo_weight
        # print("== Cian Test ==: current_elbo_weight=", current_elbo_weight)

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        # 方案1：使用加权损失
        # loss = loss.mean() * self.l_simple_weight + current_elbo_weight * loss_vlb   # original_elbo_weight=0

        # 方案2：只使用MSE损失
        loss = loss.mean()
        loss_dict.update({f'{log_prefix}/loss': loss})  # loss is weighted sum loss

        return loss, loss_dict


    # 从 Batch中获得对应key的内容
    def get_input(self, batch, k):
        if k in batch.keys():
            x = batch[k]
            if isinstance(x, torch.Tensor):
                x = x.to(memory_format=torch.contiguous_format).float()
        else:
            x = None
        return x


    def maybe_record_conditions(self, batch):
        """Gather conditions information over the full dataset in the first
        training epoch.
        """
        conditions = self.get_input(batch, self.cond_key)  # self.cond_key = cond
        if (self.current_epoch == 0) and (conditions is not None):
            self.cond_names = list(conditions)
            conditions_tensor = torch.cat([as_1d_vec(conditions[k]) for k in self.cond_names], dim=1)
            # FIX: option to skip (load from pre-trained weights)
            if self.unique_conditions is not None and conditions_tensor.shape[1] != self.unique_conditions.shape[1]:
                self.unique_conditions = conditions_tensor.unique(dim=0)
            else:
                self.unique_conditions = (
                    conditions_tensor.unique(dim=0)
                    if self.unique_conditions is None
                    else torch.cat((self.unique_conditions, conditions_tensor)).unique(dim=0)
                )

    def shared_step(self, batch):  # batch是输入数据
        x = self.get_input(batch, self.input_key)  # x = batch["input"] 也是重建目标，用来计算损失函数的
        conditions = self.get_input(batch, self.cond_key)
        input_gene_list = self.get_input(batch, self.input_gene_list_key)
        text_embeddings = self.get_input(batch, self.text_embeddings_key)
        aug_graph = self.get_input(batch, self.aug_graph_key)
        target = self.get_input(batch, self.target_key)  # "target"   训练数据没有这个属性，所以是None

        ''' 调用自己self（EMA ScDiff模型），开始训练 '''
        loss, loss_dict = self(x, conditions=conditions, input_gene_list=input_gene_list,
                               text_embeddings=text_embeddings,
                               aug_graph=aug_graph, target=target)
        return loss, loss_dict


    def training_step(self, batch, batch_idx):
        self.maybe_record_conditions(batch)
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        log_prefix = 'train' if self.training else 'val'
        print()
        print("=======================================================================================================================")
        print(f"Epoch{self.current_epoch}")
        print("Training: loss_sum:", format(loss, '.3f'), "loss_MSE:", format(loss_dict[f'{log_prefix}/loss_MSE'], '.3f'),
              "loss_vlb:", format(loss_dict[f'{log_prefix}/loss_vlb'], '.3f'))
        if self.use_scheduler:  # No
            # 在 PyTorch Lightning 框架中，self.optimizers() 方法用于在您的 LightningModule 内部获取在 configure_optimizers 方法中定义和返回的优化器。
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss


    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)


    @torch.no_grad()   # 验证不需要反向传播，训练才需要
    def validation_step(self, batch, batch_idx):
        loss, loss_dict_no_ema = self.shared_step(batch)
        # print("Validation Step - Loss Dict without EMA:", loss_dict_no_ema)

        with self.ema_scope():  # 将模型参数替换为EMA参数，这一替换只在 ema_scope 上下文管理器的作用域内有效，即在这个上下文管理器代码块内部的计算中使用 EMA 参数。
            loss, loss_dict_ema = self.shared_step(batch)
            # eg. loss_MSE + ema = loss_MSE_ema(loss_simple_ema)    loss_vlb_ema  loss_ema
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            print("Validation: Loss with EMA:", loss_dict_ema)

        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log('val/loss_ema', loss_dict_ema['val/loss_ema'], on_epoch=True, prog_bar=True)


    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=None):
        extras = self.get_input(batch, self.extras_key) or {}
        if (
            (de_gene_idx_dict := extras.get("rank_genes_groups_cov_all_idx_dict")) is None
            or (ndde20_idx_dict := extras.get("top_non_dropout_de_20")) is None
        ):
            return


    '''
    实现LightningModule的 test_step方法 （每次只批次执行一次）
    '''
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        print("== Cian Test ==: 进行test之前：batch.keys:", batch.keys())
        # dict_keys(['input', 'cond', 'mask', 'masked_target'])
        # for key, value in batch.items():
        #     print("Key:", key)
        #     print("Value:", value)
        #     if isinstance(value, torch.Tensor):
        #         print("shape:", value.shape)
        #     else:
        #         print("batch shape:", value['batch'].shape)
        #         print("celltype shape:", value['cell_type'].shape)

        x = self.get_input(batch, self.input_key)  # input_key = "input"
        print("test step 输入的数据中非零元素总数：",  torch.count_nonzero(x) )
        conditions = self.get_input(batch, self.cond_key)  # "cond"
        # print("== Cian Test ==: conditions is ", conditions)
        input_gene_list = self.get_input(batch, self.input_gene_list_key)
        text_embeddings = self.get_input(batch, self.text_embeddings_key)
        print("== Cian Test ==: text_embeddings is ", text_embeddings)
        aug_graph = self.get_input(batch, self.aug_graph_key)
        denoise_mask = self.get_input(batch, self.denoise_mask_key)  # "mask"  adata.layers['test_mask']  其实是test_mask!!

        extras = self.get_input(batch, self.extras_key)
        de_gene_idx_dict = None if extras is None else extras.get("rank_genes_groups_cov_all_idx_dict")
        ndde20_idx_dict = None if extras is None else extras.get("top_non_dropout_de_20")

        null_conditions = {i: torch.zeros_like(j) for i, j in conditions.items()}
        raw_x = self.get_input(batch, self.raw_input_key)
        denoise_t_sample = torch.tensor([self.denoise_t_sample], dtype=torch.long, device=x.device)  # 1000
        denoise_mask = denoise_mask.bool()

        '''  使用EMA参数imputation  '''
        with self.ema_scope():
            print(f"从随机噪音开始扩散生成{self.num_timesteps}次...")
            # TODO: 正常来说1000次降噪是这样的  denoise_mask 其实是 test_mask
            # 参数1：ctxt=x
            denoise_recon = self.sample(x, denoise_t_sample, conditions, input_gene_list,
                                        text_embeddings, aug_graph=aug_graph).cpu()
            denoise_recon.clamp_(0)

        denoise_target = self.get_input(batch, self.denoise_target_key).cpu()  # "masked_target" = "target" 完整的没有缺失的数据（归一化了）
        denoise_mask = denoise_mask.cpu()

        ###########################################################################
        path = self.save_path
        # save_result(x, path, "x(normalized masked input)")  # 有缺失，有归一化
        save_result(raw_x, path, "raw_x")   # 无缺失，无归一化
        save_result(denoise_recon, path, "denoise_recon")
        save_result(denoise_target, path, "denoise_target")  # 无缺失，有归一化
        # save_result(denoise_mask, path, "denoise_mask")
        # save_result(latent, path, "latent_embeds")   # 512维度embeddings 提取的细胞特征，用于聚类

        # depth_factor = cal_depth_factor(raw_x)
        denoise_recon_inv = inverse_scHiC_normalize(denoise_recon)
        save_result(denoise_recon_inv, path, "denoise_recon_inv")


        out = {
            'x': x.cpu(),
            'raw_x': raw_x.cpu() if raw_x is not None else None,  # 2. None
            'target': None,
            'denoise_mask': denoise_mask,
            'denoise_recon': denoise_recon,  # 扩散模型预测的X0
            'denoise_target': denoise_target,  # 原始完整数据  和1一样
            'conditions': {k: conditions[k].cpu() for k in sorted(conditions)},
            'de_gene_idx_dict': de_gene_idx_dict,
            'ndde20_idx_dict': ndde20_idx_dict,
        }
        # 结果存起来以便在on_test_epoch_end里访问
        self.test_step_outputs.append(out)

        return out


    '''  这个方法在整个测试数据集被遍历一次之后调用，用于执行一些在测试阶段结束时需要处理的汇总工作，如计算整个测试集的平均损失、精度等。  '''
    '''  rocon: 归一化后的预测数据， recon_inv: recon经过反归一化的预测数据  '''
    '''  recon与x比较， recon_inv与raw_x比较  '''
    @torch.no_grad()
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        x = torch.cat([outdict['x'].cpu() for outdict in outputs])
        conditions = {k: torch.cat([outdict['conditions'][k].cpu() for outdict in outputs]).numpy() for k in outputs[0]['conditions'].keys()}
        # print("== Cian Test ==: on_test_epoch_end() conditions=?", conditions)
        # 用于保存test的评估指标
        metrics_dict = {}

        if self.denoise_flag:
            denoise_mask = torch.cat([outdict['denoise_mask'].cpu() for outdict in outputs])
            denoise_recon = torch.cat([outdict['denoise_recon'].cpu() for outdict in outputs])
            denoise_target = torch.cat([outdict['denoise_target'].cpu() for outdict in outputs])

            ''''  Evaluate  '''
            raw_x = torch.cat([outdict['raw_x'].cpu() for outdict in outputs])
            metrics_dict.update(denoising_eval(denoise_target, denoise_recon, raw_x))  # true pred mask

        self.log_dict(metrics_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.test_step_outputs.clear()



    '''  根据这个优化器，反向传播时进行模型参数的更新  '''
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:  # False
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        # opt = torch.optim.RMSprop(params, lr=lr, weight_decay=self.weight_decay)

        if self.use_scheduler:  # False
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt



    # 计算KL散度的损失函数，但是没用这个
    @torch.no_grad()
    def calculate_vlb(self, x_start, conditions=None, input_gene_list=None, aug_graph=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.a
        """
        device = x_start.device
        batch_size = x_start.shape[0]
        full_mask = torch.ones_like(x_start, dtype=bool)

        vlb = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            p_mean, _, p_log_variance = self.p_mean_variance(x_start, x_t, t, self.clip_denoised,
                                                             conditions, input_gene_list, aug_graph=aug_graph,
                                                             mask=full_mask)
            q_mean, _, q_log_variance = self.q_posterior(x_start, x_t, t)
            vlb.append(self.normal_kl(q_mean, q_log_variance, p_mean, p_log_variance))
        vlb = torch.stack(vlb, dim=1)
        prior_kl = self.calculat_prior_kl(x_start)
        total_vlb = vlb.sum(dim=1) + prior_kl
        return total_vlb

    def calculat_prior_kl(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = self.normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    ''' 计算两个高斯分布之间的KL散度 '''
    ''' loss_vlb: 这部分是正则化项，用于度量模型学习的潜在表示的分布与先验分布之间的差异。 '''
    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    # ''' Cian: 动态调整loss_vlb在总loss中的占比，不然一直是0  '''
    # def vlb_loss_weight_linear_scheduler(self, current_epoch, max_epochs, start_weight, end_weight):
    #     return start_weight + (end_weight - start_weight) * (current_epoch / max_epochs)

    def vlb_loss_weight_expon_scheduler(self, current_epoch, max_weight=0.3, start_weight=0.1):
        initial_weight = 0.01
        max_weight = 0.3
        max_epochs = 3000

        gamma = (max_weight / initial_weight) ** (1 / max_epochs)
        print(f"Calculated gamma: {gamma}")

        return start_weight * (gamma ** current_epoch)



    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)


