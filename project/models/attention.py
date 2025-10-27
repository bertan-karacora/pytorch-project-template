import logging
import math

import torch


_LOGGER = logging.getLogger(__name__)


def encode_position_1d(positions, shape, device):
    term = torch.exp(torch.arange(0, shape[1], 2, dtype=torch.float, device=device) * -(math.log(10000.0) / shape[1]))

    encoding_position = torch.zeros(shape[0], shape[1], device=device)
    encoding_position[:, 0::2] = torch.sin(positions * term)
    encoding_position[:, 1::2] = torch.cos(positions * term)

    return encoding_position


def encode_position_3d(positions, num_channels, device):
    num_channels_each = num_channels // 3
    num_channels_half = num_channels_each // 2

    encodings = []
    for i in range(3):
        pos = positions[:, :, i][..., None]
        term = torch.exp(torch.arange(0, num_channels_half, dtype=torch.float, device=device) * -(math.log(10000.0) / num_channels_half))
        sin = torch.sin(pos * term)
        cos = torch.cos(pos * term)
        encodings.append(torch.cat([sin, cos], dim=-1))
    encodings = torch.cat(encodings, dim=-1)

    return encodings


class SelfAttention(torch.nn.Module):
    def __init__(self, num_channels_in, num_heads, prob_dropout=0.1, use_batch_first=False):
        super().__init__()

        self.module_attention = None
        self.module_dropout = None
        self.module_norm = None
        self.num_channels_in = num_channels_in
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout
        self.use_batch_first = use_batch_first

        self._init()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _init(self):
        self.module_attention = torch.nn.MultiheadAttention(self.num_channels_in, self.num_heads, dropout=self.prob_dropout, batch_first=self.use_batch_first)
        self.module_norm = torch.nn.LayerNorm(self.num_channels_in)
        self.module_dropout = torch.nn.Dropout(self.prob_dropout)

    def forward(self, inpt, mask_attention=None, mask_padding_key=None):
        output = self.module_attention(inpt, inpt, value=inpt, attn_mask=mask_attention, key_padding_mask=~mask_padding_key)[0]
        output = inpt + self.module_dropout(output)
        output = self.module_norm(output)

        return output


class CrossAttention(torch.nn.Module):
    def __init__(self, num_channels_in, num_heads, prob_dropout=0.1, use_batch_first=False):
        super().__init__()

        self.module_attention = None
        self.module_dropout = None
        self.module_norm = None
        self.num_channels_in = num_channels_in
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout
        self.use_batch_first = use_batch_first

        self._init()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _init(self):
        self.module_attention = torch.nn.MultiheadAttention(self.num_channels_in, self.num_heads, dropout=self.prob_dropout, batch_first=self.use_batch_first, add_zero_attn=True)
        self.module_norm = torch.nn.LayerNorm(self.num_channels_in)
        self.module_dropout = torch.nn.Dropout(self.prob_dropout)

    def forward(self, inpt_query, inpt_key, mask_attention=None, mask_padding_key=None):
        output = self.module_attention(inpt_query, inpt_key, value=inpt_key, attn_mask=mask_attention, key_padding_mask=~mask_padding_key)[0]
        output = inpt_query + self.module_dropout(output)
        output = self.module_norm(output)

        return output


class MultiHeadAttentionSpatial(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, batch_first=True, spatial_multihead=True, spatial_dim=5):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim

        self.w_qs = torch.nn.Linear(d_model, d_model)
        self.w_ks = torch.nn.Linear(d_model, d_model)
        self.w_vs = torch.nn.Linear(d_model, d_model)

        self.fc = torch.nn.Linear(d_model, d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        self.lang_cond_fc = torch.nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None):
        residual = q

        # (b, l, d_model) -> (b, l, n_head, d_k) -> (n_head, b, l, d_k)
        q = self.w_qs(q).view(q.size(0), q.size(1), self.n_head, -1).permute(2, 0, 1, 3)
        k = self.w_ks(k).view(k.size(0), k.size(1), self.n_head, -1).permute(2, 0, 1, 3)
        v = self.w_vs(v).view(v.size(0), v.size(1), self.n_head, -1).permute(2, 0, 1, 3)

        attn = torch.einsum("hblk,hbtk->hblt", q, k) / torch.sqrt(q.shape[-1])

        # spatial weights
        spatial_weights = self.lang_cond_fc(residual)  # (b, l, spatial_n_head * (spatial_dim+1))
        spatial_weights = spatial_weights.view(spatial_weights.size(0), spatial_weights.size(1), self.spatial_n_head, self.spatial_dim + 1).permute(2, 0, 1, 3)  # (h, b, l, d)

        spatial_bias = spatial_weights[..., :1]  # (h, b, l, 1)
        spatial_weights = spatial_weights[..., 1:]  # (h, b, l, d)

        loc_attn = torch.einsum("hbld,bltd->hblt", spatial_weights, pairwise_locs) + spatial_bias
        loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            # (b, t) -> (h, b, l, t)
            mask = key_padding_mask.unsqueeze(0).unsqueeze(2).expand(self.n_head, -1, q.size(2), -1)
            attn = attn.masked_fill(mask, -torch.Tensor(float("Inf")))
            loc_attn = loc_attn.masked_fill(mask, 0)

        fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
        fused_attn = torch.softmax(fused_attn, dim=3)

        assert torch.sum(torch.isnan(fused_attn)) == 0, print(fused_attn)

        output = torch.einsum("hblt,hbtv->hblv", fused_attn, v)
        # (n_head, b, l, d_v) -> (b, l, n_head * d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(output.size(1), output.size(2), -1)

        output = self.fc(output)

        return output, fused_attn


class SelfAttentionSpatial(torch.nn.Module):
    def __init__(self, num_channels_in, num_heads, prob_dropout=0.1, use_batch_first=False):
        super().__init__()

        self.module_attention = None
        self.module_dropout = None
        self.module_norm = None
        self.num_channels_in = num_channels_in
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout
        self.use_batch_first = use_batch_first

        self._init()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _init(self):
        self.module_attention = MultiHeadAttentionSpatial(self.num_channels_in, self.num_heads, dropout=self.prob_dropout, batch_first=self.use_batch_first)
        self.module_norm = torch.nn.LayerNorm(self.num_channels_in)
        self.module_dropout = torch.nn.Dropout(self.prob_dropout)

    def forward(self, inpt, pairwise_locs, mask_attention=None, mask_padding_key=None):
        output = self.module_attention(inpt, inpt, inpt, pairwise_locs=pairwise_locs, key_padding_mask=~mask_padding_key)[0]
        output = inpt + self.module_dropout(output)
        output = self.module_norm(output)

        return output
