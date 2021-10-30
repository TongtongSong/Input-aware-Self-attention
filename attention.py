class InputAwareSelfAttention(nn.Module):
    """InputAwareSelfAttention.
    Args:
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_feat: int, dropout_rate: float):
        super(InputAwareSelfAttention, self).__init__()
        self.linear_q = nn.Linear(2*n_feat, n_feat, bias=False)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=False)
        self.embedding_out = nn.Linear(n_feat*2, n_feat, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor,value: torch.Tensor
    ):
        mean = query.mean(dim=1,keepdim=True)
        std = torch.sqrt(query.var(dim=1,unbiased=False, keepdim=True).clamp(min=1.0e-10))
        mean_std = torch.cat((mean,std),dim=2)
        q = self.linear_q(mean_std) # B x 1 x D

        k = self.linear_k(key) # B x T x D
        v = value # B x T x D
        return q, k, v

    def mask_score(self,
        scores: torch.Tensor, mask: Optional[torch.Tensor]
    ):
        if mask is not None:
            mask = mask.eq(0)  # (batch, 1, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        return p_attn

    def forward(self,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: Optional[torch.Tensor]
    ):
        q, k, v = self.forward_qkv(query, key, value)

        score = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(k.shape[2]) # B x 1 x T
        score = self.mask_score(score, mask) # B x 1 x T
        att_mean = torch.matmul(score, v) # B x 1 x D
        att_stddev = torch.sqrt(torch.matmul(score, (v - att_mean) ** 2).clamp(min=1.0e-10)) # B x 1 x D
        xs = self.linear_out(att_mean)
        embedding = self.embedding_out(torch.cat((att_mean, att_stddev),dim=-1)).squeeze(1)
        return  xs, embedding
