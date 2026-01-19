import torch
import torch.nn as nn
import torch.nn.functional as F

from scdiff.utils.modules import create_activation, create_norm

'''  将输入的基因表达数据嵌入到一个连续的向量空间中（512）  '''
'''  out = torch.sparse.mm(x, feat)         '''
'''  linear mapping 线性映射                  '''

# embed_dim: 512
# decoder_embed_dim: 512  为什么还是512呢？


class Embedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hidden, norm, activation='gelu', dropout=0.,
                 gene_emb=None, fix_embedding=False):
        super().__init__()

        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = {j: i for i, j in enumerate(pretrained_gene_list)}

        if gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
        else:
            num_genes = len(pretrained_gene_list)  # 10731
            ''' self.emb: embed参数  形状[num_genes, num_hidden] '''
            # 1. trch.rand() 利用randn函数创建一个形状为（10731x512）的张量，全都是来自高斯分布的随机数，*0.005是进行缩放，最小
            # 2. nn.Parameter 包装这个张量，使其成为模型的一个可训练参数。
            # （cell_num, 10731) x (10731, 512) = (cell_num, 512)

            # 标准正态分布的 99.7% 数据范围在均值的 3 个标准差之内。因此：
            # 在未缩放前，大部分数值会在 [-3, 3] 之间。
            # 缩放后，大部分数值会在 [-3 * 0.005, 3 * 0.005] 之间，即 [-0.015, 0.015] 之间。
            self.emb = nn.Parameter(torch.randn([num_genes, num_hidden], dtype=torch.float32) * 0.005)

        if fix_embedding:
            self.emb.requires_grad = False

        self.post_layer = nn.Sequential(
            create_activation(activation),  # GELU
            create_norm(norm, num_hidden),  # layernorm
            nn.Dropout(dropout),  # dropout=0
        )

    def forward(self, x, input_gene_list=None, input_gene_idx=None):
        if input_gene_idx is not None:
            gene_idx = input_gene_idx
        elif input_gene_list is not None:
            gene_idx = torch.tensor([self.gene_index[o] for o in input_gene_list if o in self.gene_index]).long()
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError(
                    'The input gene size is not the same as the pretrained gene list. '
                    'Please provide the input gene list.',
                )
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)

        feature = F.embedding(gene_idx, self.emb)
        out = torch.sparse.mm(x, feature)
        out = self.post_layer(out)

        return out, gene_idx
