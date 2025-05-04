import torch
import torchvision.transforms.v2 as tv_transforms

from self_supervised_learning_of_depth_and_motion.models.cnn import BlockConv2d, BlockConvTranspose2d


class GeneratorDCGAN2d(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, nums_channels_hidden, use_normalize=False, kwargs_normalize=None):
        super().__init__()

        self.body = None
        self.kwargs_normalize = kwargs_normalize
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.use_normalize = use_normalize

        self._init()

    def _init(self):
        nums_channels_body = [self.num_channels_in] + list(self.nums_channels_hidden) + [self.num_channels_out]

        modules_body = []
        for i, (num_channels_i, num_channels_o) in enumerate(zip(nums_channels_body[:-1], nums_channels_body[1:])):
            modules_body += [
                BlockConvTranspose2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **dict(
                        shape_kernel_conv=(4, 4),
                        kwargs_conv=dict(
                            padding=1,
                            stride=1 if i == 0 else 2,
                        ),
                        name_layer_norm="BatchNorm2d" if i != len(nums_channels_body) - 2 else None,
                        name_layer_act="ReLU" if i != len(nums_channels_body) - 2 else "Tanh",
                    ),
                )
            ]
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, input, *args):
        output = input.view(*input.shape, 1, 1)
        output = self.body(output)

        # Map to [0, 1]
        output = output * 0.5 + 0.5

        if self.use_normalize:
            output = tv_transforms.functional.normalize(output, **self.kwargs_normalize)

        return output

    def sample_latent(self, num_samples=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        samples = torch.randn(num_samples, self.num_channels_in, device=device)
        return samples

    def decode(self, input, *args):
        return self(input)

    @torch.no_grad()
    def sample(self, num_samples=16):
        code_latent = self.sample_latent(num_samples)
        samples = self(code_latent)
        return samples


class DiscriminatorDCGAN2d(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, nums_channels_hidden, prob_dropout=None):
        super().__init__()

        self.body = None
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        nums_channels_body = [self.num_channels_in] + list(self.nums_channels_hidden) + [self.num_channels_out]

        modules_body = []
        for i, (num_channels_i, num_channels_o) in enumerate(zip(nums_channels_body[:-1], nums_channels_body[1:])):
            modules_body += [
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **dict(
                        shape_kernel_conv=(4, 4),
                        kwargs_conv=dict(
                            padding=1,
                            stride=2 if i != len(nums_channels_body) - 2 else 4,
                        ),
                        name_layer_norm="BatchNorm2d" if i != len(nums_channels_body) - 2 else None,
                        name_layer_act="LeakyReLU" if i != len(nums_channels_body) - 2 else "Sigmoid",
                        kwargs_act=dict(negative_slope=0.2) if i != len(nums_channels_body) - 2 else None,
                        prob_dropout=self.prob_dropout if i != len(nums_channels_body) - 2 else None,
                    ),
                )
            ]
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, input, *args):
        output = self.body(input)
        output = output.view(-1)
        return output


class GeneratorCDCGAN2d(torch.nn.Module):
    def __init__(self, num_channels_in_latent, num_channels_in_label, num_channels_out, nums_channels_hidden, use_normalize=False, kwargs_normalize=None):
        super().__init__()

        self.body = None
        self.kwargs_normalize = kwargs_normalize
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_in_label = num_channels_in_label
        self.num_channels_in_latent = num_channels_in_latent
        self.num_channels_out = num_channels_out
        self.tail_label = None
        self.tail_latent = None
        self.use_normalize = use_normalize

        self._init()

    def _init(self):
        nums_channels_body = list(self.nums_channels_hidden) + [self.num_channels_out]

        self.tail_latent = BlockConvTranspose2d(
            num_channels_in=self.num_channels_in_latent,
            num_channels_out=nums_channels_body[0] // 2,
            **dict(
                shape_kernel_conv=(4, 4),
                kwargs_conv=dict(
                    padding=1,
                    stride=1,
                ),
                name_layer_norm="BatchNorm2d",
                name_layer_act="ReLU",
            ),
        )

        self.tail_label = BlockConvTranspose2d(
            num_channels_in=self.num_channels_in_label,
            num_channels_out=nums_channels_body[0] // 2,
            **dict(
                shape_kernel_conv=(4, 4),
                kwargs_conv=dict(
                    padding=1,
                    stride=1,
                ),
                name_layer_norm="BatchNorm2d",
                name_layer_act="ReLU",
            ),
        )

        modules_body = []
        for i, (num_channels_i, num_channels_o) in enumerate(zip(nums_channels_body[:-1], nums_channels_body[1:])):
            modules_body += [
                BlockConvTranspose2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **dict(
                        shape_kernel_conv=(4, 4),
                        kwargs_conv=dict(
                            padding=1,
                            stride=2,
                        ),
                        name_layer_norm="BatchNorm2d" if i != len(nums_channels_body) - 2 else None,
                        name_layer_act="ReLU" if i != len(nums_channels_body) - 2 else "Tanh",
                    ),
                )
            ]
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, input, label):
        output_latent = input.view(*input.shape, 1, 1)
        output_latent = self.tail_latent(output_latent)

        output_label = label.float()
        output_label = output_label.view(output_label.shape[0], 1, 1, 1)
        output_label = self.tail_label(output_label)

        output = torch.concat((output_latent, output_label), dim=1)
        output = self.body(output)

        # Map to [0, 1]
        output = output * 0.5 + 0.5

        if self.use_normalize:
            output = tv_transforms.functional.normalize(output, **self.kwargs_normalize)

        return output

    def decode(self, input, label):
        return self(input, label)

    def sample_latent(self, num_samples=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        samples = torch.randn(num_samples, self.num_channels_in_latent, device=device)
        return samples

    @torch.no_grad()
    def sample(self, label, num_samples=16):
        code_latent = self.sample_latent(num_samples)
        samples = self(code_latent, label)
        return samples


class DiscriminatorCDCGAN2d(torch.nn.Module):
    def __init__(self, num_channels_in_features, num_channels_in_label, num_channels_out, nums_channels_hidden, prob_dropout=None):
        super().__init__()

        self.body = None
        self.nums_channels_hidden = nums_channels_hidden
        self.num_channels_in_label = num_channels_in_label
        self.num_channels_in_features = num_channels_in_features
        self.num_channels_out = num_channels_out
        self.prob_dropout = prob_dropout
        self.tail_label = None
        self.tail_features = None

        self._init()

    def _init(self):
        nums_channels_body = list(self.nums_channels_hidden) + [self.num_channels_out]

        self.tail_features = BlockConv2d(
            num_channels_in=self.num_channels_in_features,
            num_channels_out=nums_channels_body[0] // 2,
            **dict(
                shape_kernel_conv=(4, 4),
                kwargs_conv=dict(
                    padding=1,
                    stride=2,
                ),
                name_layer_norm="BatchNorm2d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.2),
                prob_dropout=self.prob_dropout,
            ),
        )

        self.tail_label = BlockConv2d(
            num_channels_in=self.num_channels_in_label,
            num_channels_out=nums_channels_body[0] // 2,
            **dict(
                shape_kernel_conv=(4, 4),
                kwargs_conv=dict(
                    padding=1,
                    stride=2,
                ),
                name_layer_norm="BatchNorm2d",
                name_layer_act="LeakyReLU",
                kwargs_act=dict(negative_slope=0.2),
                prob_dropout=self.prob_dropout,
            ),
        )

        modules_body = []
        for i, (num_channels_i, num_channels_o) in enumerate(zip(nums_channels_body[:-1], nums_channels_body[1:])):
            modules_body += [
                BlockConv2d(
                    num_channels_in=num_channels_i,
                    num_channels_out=num_channels_o,
                    **dict(
                        shape_kernel_conv=(4, 4),
                        kwargs_conv=dict(
                            padding=1,
                            stride=2 if i != len(nums_channels_body) - 1 else 4,
                        ),
                        name_layer_norm="BatchNorm2d" if i != len(nums_channels_body) - 2 else None,
                        name_layer_act="LeakyReLU" if i != len(nums_channels_body) - 2 else "Sigmoid",
                        kwargs_act=dict(negative_slope=0.2) if i != len(nums_channels_body) - 2 else None,
                        prob_dropout=self.prob_dropout if i != len(nums_channels_body) - 2 else None,
                    ),
                )
            ]
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, input, label):
        output_features = self.tail_features(input)

        output_label = label.float()
        output_label = output_label.view(output_label.shape[0], 1, 1, 1)
        output_label = output_label.repeat(1, 1, *input.shape[2:])
        output_label = self.tail_label(output_label)

        output = torch.concat((output_features, output_label), dim=1)
        output = self.body(output)
        output = output.view(-1)

        return output
