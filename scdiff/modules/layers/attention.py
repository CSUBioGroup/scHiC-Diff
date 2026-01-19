from typing import Optional

import torch
from torch import nn, einsum  # einsum 是一个强大的函数，用于根据指定的下标表达式来计算张量操作。主要用来求内积 in sum
from einops import rearrange, repeat
import torch.nn.functional as F

from scdiff.modules.layers.basic import FeedForward
from scdiff.utils.misc import default, exists, max_neg_value
from scdiff.utils.modules import BatchedOperation, create_norm, zero_module

# 多头注意力就是在一个输入序列上使用多个自注意力机制，得到多组注意力结果，然后将这些结果进行拼接和线性投影得到最终输出。
# query_dim=512, qkv_bias=True
# input的形状： batch_size, seq_len, feature_dim
# 通过多个注意力头，模型可以从不同的子空间中提取特征，从而捕捉到数据中更多的模式和关系。

'''  具体调用： 
         dim=512, n_heads=8, d_head=64, self_attn=False, cross_attn=True, context_dim=512, 
         qkv_bias=True, dropout=0, final_act=None, gated_ff=True

    input: (cell_num, num_tokens, 512) -> output: (cell_num, num_tokens, 512)
'''
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, num_heads=8, dim_head=64, dropout=0., qkv_bias=False):
        super().__init__()
        self.dim_head = dim_head
        inner_dim = dim_head * num_heads   # 64x8 = 512
        context_dim = default(context_dim, query_dim)  # context_dim=query_dim=512

        # 计算 scale 作为 dim 的平方根的倒数。这个缩放因子用来防止查询和键之间的点积变得过大从而softmax函数的梯度消失。
        self.scale = dim_head ** -0.5  # 1/8 = 0.125
        self.num_heads = num_heads  # 8

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)    # 512 to 512
        self.to_k = nn.Linear(context_dim, inner_dim, bias=qkv_bias)  # 512 to 512
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)  # 512 to 512

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),   # 512 to 512
            nn.Dropout(dropout)
        )

    # 调用： x = blk(x, context)
    def forward(self, x, *, context=None, mask=None):
        # print("== Cian test ==   context:", context)
        h = self.num_heads   # 8
        b, n, _ = x.size()
        d = self.dim_head

        ''' 1. 注意： context=None时，交叉注意力机制退化为自注意力机制  '''
        context = default(context, x)

        # x含有时间步信息，context是没被mask的上下文
        q = self.to_q(x).view(b, n, h, d).permute(0, 2, 1, 3)
        k = self.to_k(context).view(b, n, h, d).permute(0, 2, 3, 1)
        v = self.to_v(context).view(b, n, h, d).permute(0, 2, 1, 3)

        sim = torch.matmul(q, k) * self.scale
        attention = F.softmax(sim, dim=-1)

        out = torch.matmul(attention, v).permute(0,2,1,3).contiguous().view(b, n, -1)  # -1表示这个维度自动计算
        return self.to_out(out)  # batch_size, 1, 512


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int = 64,
        self_attn: bool = True,
        cross_attn: bool = False,
        ts_cross_attn: bool = False,
        final_act: Optional[nn.Module] = None,
        dropout: float = 0.,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        qkv_bias: bool = False,
        linear_attn: bool = False,
    ):
        super().__init__()
        assert self_attn or cross_attn, 'At least on attention layer'
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        # FeedForward(dim=512, dropout=0, glu=True)   512  —>512x4 -> 512
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        if ts_cross_attn:
            raise NotImplementedError("Deprecated, please remove.")  # FIX: remove ts_cross_attn option
            # assert not (self_attn or linear_attn)
            # attn_cls = TokenSpecificCrossAttention
        else:
            assert not linear_attn, "Performer attention not setup yet."  # FIX: remove linear_attn option
            attn_cls = CrossAttention   # attention_class is CrossAttention Class

        if self.cross_attn:
            self.attn1 = attn_cls(
                query_dim=dim,
                context_dim=context_dim,
                num_heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )  # is self-attn if context is none
        if self.self_attn:
            self.attn2 = attn_cls(
                query_dim=dim,
                num_heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )  # is a self-attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.act = final_act

    @BatchedOperation(batch_dim=0, plain_num_dim=2)
    def forward(self, x, context=None, cross_mask=None, self_mask=None, **kwargs):
        if self.cross_attn:  # attn1(x)+x  x使用自注意力机制计算后 再与 x本身相加
            x = self.attn1(self.norm1(x), context=context, mask=cross_mask, **kwargs) + x
        if self.self_attn:
            x = self.attn2(self.norm2(x), mask=self_mask, **kwargs) + x

        x = self.ff(self.norm3(x)) + x  # FeedForward
        if self.act is not None:
            x = self.act(x)
        return x


