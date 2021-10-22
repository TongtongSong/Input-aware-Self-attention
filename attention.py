import math
from typing import Optional, Tuple

import torch
from torch import nn
def StatisticPooling(feats):
    assert feats.dim() == 3 # (batch, time, dim)
    mean = feats.mean(1)
    variance = torch.mul(feats, feats).sum(1) - torch.mul(mean,mean)/frames.size(1)
    return torch.cat((mean,variance),dim=-1).unsqueeze(1)  # (batch, 1, dim)

class InputAwareSelfAttention(nn.Module):
    """InputAwareSelfAttention.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        super(InputAwareSelfAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(2*n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.embedding_out = nn.Linear(2*n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor,value: torch.Tensor
    ):
        n_batch = query.size(0)
        query = StatisticPooling(query) # (batch, 1, size*2)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = value.view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, 1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def mask_score(self,
        scores: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        return p_attn

    def forward(self,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: Optional[torch.Tensor]
    ):
        n_batch = query.size(0)
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) #(batch, head, 1, time2)
        scores = self.mask_score(scores, mask)
        scores = scores.squeeze(2).unsqueeze(-1)
        attn_out = scores*v # (batch, head, time2, d_k)
        mean = attn_out.sum(2).view(n_batch, self.h * self.d_k)
        variance = torch.mul(v, attn_out).sum(2).view(n_batch, self.h * self.d_k) - torch.mul(mean,mean)
        return  self.linear_out(mean).unsqueeze(1), self.embedding_out(torch.cat((mean,variance),dim=-1))
