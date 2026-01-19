import torch
import torch.nn as nn

from scdiff.modules.layers.attention import BasicTransformerBlock
from scdiff.modules.layers.basic import FeedForward
from scdiff.utils.diffusion import ConditionEncoderWrapper
from scdiff.utils.modules import create_norm

'''  具有 cross-attention 
     调用： self.encoder = Encoder(depth=6, decoder_embed_dim=512, decoder_num_heads=8, decoder_dim_head=64,
                               dropout=0, cond_type="crossattn", cond_cat_input=False)
     具有6个BasicTransformerBlock的Encoder
     input: (cell_num,512) -> output: (cell_num,512)
'''

class Encoder(nn.Module):

    def __init__(
        self,
        depth,
        dim,        # 512
        num_heads,  # 8
        dim_head,   # 64
        *,
        dropout=0.,
        cond_type='crossattn',
        cond_cat_input=False,
    ):
        super().__init__()

        self.cond_cat_input = cond_cat_input   # False

        # yep.就是这个
        if cond_type == 'crossattn':
            self.blocks = nn.ModuleList([
                # dim=512, num_heads=8, dim_head=64
                BasicTransformerBlock(dim, num_heads, dim_head, self_attn=False, cross_attn=True, context_dim=dim,
                                      qkv_bias=True, dropout=dropout, final_act=None)
                for _ in range(depth)])
        elif cond_type == 'mlp':
            self.blocks = nn.ModuleList([
                ConditionEncoderWrapper(nn.Sequential(
                    nn.Linear(dim, dim),
                    "gelu",
                    create_norm("layernorm", dim),
                    nn.Dropout(dropout),
                )) for _ in range(depth)])
        elif cond_type == 'stackffn':
            self.blocks = nn.ModuleList([
                ConditionEncoderWrapper(
                    FeedForward(dim, mult=4, glu=False, dropout=dropout)
                ) for _ in range(depth)])
        else:
            raise ValueError(f'Unknown conditioning type {cond_type!r}')

    # forward_decoder中调用：  x = self.encoder(x, context_list, cond_emb_list=None)
    def forward(self, x, context_list, cond_emb_list):
        # XXX: combine context_list and cond_emb_list in conditioner?..
        x = x.unsqueeze(1)   # 在张量 x 的（索引为1）处插入一个新的维度。 从 (cell_num, 512) 变成了 (cell_num, 1, 512)

        # blocks, context_list, cond_emb_emb 都是6层
        # context_list = [x1, x2, x3, x4, x5, x6]
        # cond_emb_list = [None] * 6 = [None, None, None, None, None, None]
        # 作用：融合context和condition 信息
        stack = zip(self.blocks, reversed(context_list), reversed(cond_emb_list))
        for i, (blk, ctxt, cond_emb) in enumerate(stack):
            full_cond_emb_list = list(filter(lambda x: x is not None, (ctxt, cond_emb)))
            # full_cond_emb_list = [(x1)]   bcz cond_emb is None
            if self.cond_cat_input:  # false
                full_cond_emb_list.append(x)
            # 连接 同一层的ctxt和cond_emb，结果还是一个张量
            # nx512  nx512 -> 2nx512
            full_cond_emb = torch.cat(full_cond_emb_list, dim=1) if full_cond_emb_list else None
            # print("full_cond_emb的形状：", full_cond_emb.shape)
            # print("x的形状：", x.shape)

            # 网络：多头交叉注意力机制，  数据:x 作为q , context作为 q v
            x = blk(x, context=full_cond_emb)

        return x.squeeze(1)   # 删除那个维度
