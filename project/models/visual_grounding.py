import logging

import torch

import project.models.attention as attention
from project.models.attention import SelfAttention, CrossAttention, SelfAttentionSpatial
import project.models.bert as bert
from project.models.mlp import MLP


_LOGGER = logging.getLogger(__name__)


class ReferralLayer(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_heads, prob_dropout=0.1, use_cross_attention=True):
        super().__init__()

        self.mlp = None
        self.module_attention_cross = None
        self.module_attention_self = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout
        self.use_cross_attention = use_cross_attention

        self._init()

    def _init(self):
        self.module_attention_self = SelfAttention(self.num_channels_in, self.num_heads, prob_dropout=self.prob_dropout, use_batch_first=True)
        if self.use_cross_attention:
            self.module_attention_cross = CrossAttention(self.num_channels_in, self.num_heads, prob_dropout=self.prob_dropout, use_batch_first=True)

        self.mlp = MLP(num_channels_in=self.num_channels_in, nums_channels_hidden=self.nums_channels_hidden, num_channels_out=self.num_channels_in, name_layer_norm="LayerNorm", prob_dropout=self.prob_dropout)

    def forward(self, input):
        output = input["embeddings_object"]

        if self.module_attention_cross is not None:
            output = self.module_attention_cross(output, input["embeddings_language"], mask_padding_key=input["mask_language"])
            output = output * input["mask_object"][..., None]

        output = self.module_attention_self(output, mask_padding_key=input["mask_object"])
        output = output * input["mask_object"][..., None]

        output = self.mlp(output)
        output = torch.nn.functional.layer_norm(output, (self.num_channels_in,))

        return output


class Referral(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_layers=4, num_heads=12, prob_dropout=0.1, use_cross_attention=True, use_encoding_positions=True):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.prob_dropout = prob_dropout
        self.use_cross_attention = use_cross_attention
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.layers = torch.nn.ModuleList(
            [
                ReferralLayer(
                    num_channels_in=self.num_channels_in,
                    nums_channels_hidden=self.nums_channels_hidden,
                    num_heads=self.num_heads,
                    prob_dropout=self.prob_dropout,
                    use_cross_attention=self.use_cross_attention,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.apply(bert._init_weights)

    def forward(self, input):
        if self.use_encoding_positions:
            positions = torch.arange(input["embeddings_language"].shape[1], dtype=torch.float32, device=input["embeddings_language"].device)[:, None]
            encoding_position = attention.encode_position_1d(positions, shape=input["embeddings_language"].shape[1:], device=input["embeddings_language"].device)
            input["embeddings_language"] += encoding_position

            encoding_position = attention.encode_position_3d(input["positions"], input["embeddings_object"].shape[-1], device=input["embeddings_object"].device)
            input["embeddings_object"] += encoding_position

        output = input["embeddings_object"]
        for layer in self.layers:
            input_layer = dict(
                embeddings_object=output,
                mask_object=input["mask_object"],
                embeddings_language=input["embeddings_language"],
                mask_language=input["mask_language"],
            )
            output = layer(input_layer)

        return output


class ReferralSelfAttention(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_layers=4, num_heads=12, prob_dropout=0.1, use_encoding_positions=True):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.prob_dropout = prob_dropout
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.layers = torch.nn.ModuleList(
            [
                ReferralLayer(
                    num_channels_in=self.num_channels_in,
                    nums_channels_hidden=self.nums_channels_hidden,
                    num_heads=self.num_heads,
                    prob_dropout=self.prob_dropout,
                    use_cross_attention=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.apply(bert._init_weights)

    def forward(self, input):
        if self.use_encoding_positions:
            positions = torch.arange(input["embeddings_language"].shape[1], dtype=torch.float32, device=input["embeddings_language"].device)[:, None]
            encoding_position = attention.encode_position_1d(positions, shape=input["embeddings_language"].shape[1:], device=input["embeddings_language"].device)
            input["embeddings_language"] += encoding_position

            encoding_position = attention.encode_position_3d(input["positions"], input["embeddings_object"].shape[-1], device=input["embeddings_object"].device)
            input["embeddings_object"] += encoding_position

        output = torch.cat((input["embeddings_object"], input["embeddings_language"]), dim=1)
        mask = torch.cat((input["mask_object"], input["mask_language"]), dim=1)

        for layer in self.layers:
            input_layer = dict(embeddings_object=output, mask_object=mask)
            output = layer(input_layer)

        output = torch.split(output, [input["embeddings_object"].shape[1], input["embeddings_language"].shape[1]], dim=1)[0]

        return output


class ReferralLayerSpatial(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_heads, prob_dropout=0.1):
        super().__init__()

        self.mlp = None
        self.module_attention_self = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        self.module_attention_self = SelfAttentionSpatial(self.num_channels_in, self.num_heads, prob_dropout=self.prob_dropout, use_batch_first=True)
        self.mlp = MLP(num_channels_in=self.num_channels_in, nums_channels_hidden=self.nums_channels_hidden, num_channels_out=self.num_channels_in, name_layer_norm="LayerNorm", prob_dropout=self.prob_dropout)

    def forward(self, input):
        output = input["embeddings_object"]

        output = self.module_attention_self(output, input["pairwise_locs"], mask_padding_key=input["mask_object"])
        output = output * input["mask_object"][..., None]

        output = self.mlp(output)
        output = torch.nn.functional.layer_norm(output, (self.num_channels_in,))

        return output


def init_weights_3DVisTA(module):
    """Initialize the weights"""
    if isinstance(module, torch.nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class ReferralSpatial(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, num_layers=4, num_heads=12, prob_dropout=0.1, use_encoding_positions=True):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.prob_dropout = prob_dropout
        self.use_encoding_positions = use_encoding_positions

        self._init()

    def _init(self):
        self.layers = torch.nn.ModuleList(
            [
                ReferralLayerSpatial(
                    num_channels_in=self.num_channels_in,
                    nums_channels_hidden=self.nums_channels_hidden,
                    num_heads=self.num_heads,
                    prob_dropout=self.prob_dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.apply(init_weights_3DVisTA)

    def forward(self, input):
        if self.use_encoding_positions:
            encoding_position = attention.encode_position_3d(input["pos"], input["embeddings_object"].shape[-1], device=input["embeddings_object"].device)
            input["embeddings_object"] += encoding_position

        pairwise_locs = input["pos"].unsqueeze(2) - input["pos"].unsqueeze(1)

        # pairwise distances
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, dim=3) + 1e-10)

        # normalize distances
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / max_dists.view(-1, 1, 1)

        # 2D pairwise distances (using first two coords)
        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, dim=3) + 1e-10)

        # final stack
        pairwise_locs = torch.stack(
            [
                norm_pairwise_dists,
                pairwise_locs[..., 2] / pairwise_dists,
                pairwise_dists_2d / pairwise_dists,
                pairwise_locs[..., 1] / pairwise_dists_2d,
                pairwise_locs[..., 0] / pairwise_dists_2d,
            ],
            dim=3,
        )

        output = input["embeddings_object"]
        for layer in self.layers:
            input_layer = dict(
                embeddings_object=output,
                mask_object=input["mask_object"],
                pairwise_locs=pairwise_locs,
            )
            output = layer(input_layer)

        return output


class HeadGroundingMLP(torch.nn.Module):
    def __init__(self, num_channels_in, nums_channels_hidden, prob_dropout=None):
        super().__init__()

        self.layers = None
        self.num_channels_in = num_channels_in
        self.nums_channels_hidden = nums_channels_hidden
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        self.mlp = MLP(
            num_channels_in=self.num_channels_in,
            nums_channels_hidden=self.nums_channels_hidden,
            num_channels_out=1,
            name_layer_norm="LayerNorm",
            kwargs_norm=dict(eps=1e-12),
            prob_dropout=self.prob_dropout,
        )

    def forward(self, input):
        output = input["embeddings"]
        output = self.mlp(output)
        output = output[..., 0]
        output = output.masked_fill_(~input["mask"], -float("inf"))

        return output
