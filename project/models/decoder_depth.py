import torch

from self_supervised_learning_of_depth_and_motion.models.cnn import BlockConv2d


class DecoderDepth(torch.nn.Module):
    def __init__(self, shape_input=[256, 128, 256], nums_channels_latent=[256, 128, 64, 32, 16], num_channels_out=1, kwargs_block={}, alpha_elu=1.0):
        super().__init__()

        self.alpha_elu = alpha_elu
        self.body = None
        self.kwargs_block = kwargs_block or {}
        self.nums_channels_latent = nums_channels_latent
        self.num_channels_out = num_channels_out
        self.shape_input = shape_input

        self._init()

    def _init(self):
        nums_channels_body = [self.shape_input[0]] + list(self.nums_channels_latent) + [self.num_channels_out]
        modules = []
        padding_mode = "reflect"
        kernel_size = 3
        for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
            # if self.use_skips and i > 0:
            #     num_ch_in += self.num_ch_enc[i - 1]
            modules += [
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    inplace=True,
                    name_layer_act="ELU",
                    kwargs_act=dict(alpha=self.alpha_elu),
                    **self.kwargs_block,
                ),
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    inplace=True,
                    name_layer_act="ELU",
                    **self.kwargs_block,
                ),
            ]

        for num_channels_i, num_channels_o in zip(nums_channels_body[:-1], nums_channels_body[1:]):
            modules += [
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    inplace=True,
                    name_layer_act="ELU",
                    **self.kwargs_block,
                ),
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    inplace=True,
                    name_layer_act="ELU",
                    **self.kwargs_block,
                ),
            ]

        self.body = torch.nn.Sequential(*modules)

        self.head = torch.nn.Sequential(
            BlockConv2d(
                num_channels_in=num_channels_i,
                num_channels_out=num_channels_o,
                inplace=True,
                **self.kwargs_block,
            ),
            torch.nn.Sigmoid(inplace=True),
        )

    def forward_single(self, input):
        output = self.backbone(input)
        output = self.head(output)
        return output

    def forward(self, input):
        anchor = input["anchor"]
        positive = input["positive"]
        negative = input["negative"]

        inputs = torch.cat([anchor, positive, negative], dim=0)
        outputs = self.forward_single(inputs)
        output_anchor, output_positive, output_negative = torch.chunk(outputs, 3, dim=0)

        output = dict(anchor=output_anchor, positive=output_positive, negative=output_negative)
        return output


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix"""
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix"""
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T"""

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
