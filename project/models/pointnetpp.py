import logging

import torch

import project.models.bert as bert

_LOGGER = logging.getLogger(__name__)


def break_up_pc(pc):
    """
    Split the pointcloud into xyz positions and features tensors.
    This method is taken from VoteNet codebase (https://github.com/facebookresearch/votenet)

    @param pc: pointcloud [N, 3 + C]
    :return: the xyz tensor and the feature tensor
    """
    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
    return xyz, features


class PointNetPP(torch.nn.Module):
    """
    Pointnet++ encoder.
    For the hyper parameters please advise the paper (https://arxiv.org/abs/1706.02413)
    """

    def __init__(self, sa_n_points: list, sa_n_samples: list, sa_radii: list, sa_mlps: list, bn=True, use_xyz=True):
        super().__init__()

        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError("Lens of given hyper-params are not compatible")

        self.encoder = torch.nn.ModuleList()

        for i in range(n_sa):
            self.encoder.append(
                PointnetSAModule(
                    npoint=sa_n_points[i],
                    nsample=sa_n_samples[i],
                    radius=sa_radii[i],
                    mlp=sa_mlps[i],
                    bn=bn,
                    use_xyz=use_xyz,
                )
            )

        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.fc = torch.nn.Linear(out_n_points * sa_mlps[-1][-1], sa_mlps[-1][-1])

    def forward(self, features):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        xyz, features = break_up_pc(features)
        for i in range(len(self.encoder)):
            xyz, features = self.encoder[i](xyz, features)

        return self.fc(features.view(features.size(0), -1))


class EncoderPointsPointNetPP(torch.nn.Module):
    def __init__(self, prob_dropout=0.1):
        super().__init__()

        self.backbone = None
        self.dropout = None
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        self.backbone = PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
        )

        if self.prob_dropout > 0.0:
            self.dropout = torch.nn.Dropout(self.prob_dropout)

        self.apply(bert._init_weights)

        for layer in self.backbone.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()

    def load_weights(self, path):
        state_dict_pretrained = torch.load(path)
        state_dict = {}
        for k, v in state_dict_pretrained.items():
            if k[0] in ["0", "2", "4"]:  # key mapping for voxel
                k = "cls_head." + k
            k = k.replace("vision_encoder.vis_cls_head.", "cls_head.")  # key mapping for mv
            k = k.replace("point_cls_head.", "cls_head.")  # key mapping for pc
            k = k.replace("point_feature_extractor.", "backbone.")
            state_dict[k] = v

        self.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def forward(self, input):
        shape = input.shape

        output = input.reshape(shape[0] * shape[1], shape[2], shape[3])
        output = self.backbone(output)
        output = output.reshape(shape[0], shape[1], output.shape[1])

        if self.dropout is not None:
            output = self.dropout(output)

        return output
