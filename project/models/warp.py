import torch

import self_supervised_learning_of_depth_and_motion.libs.utils_import as utils_import

# https://discuss.pytorch.org/t/how-to-warp-the-image-with-optical-flow-and-grid-sample/71531/2


class WarpFlow(torch.Module):
    def __init__(
        self,
        num_channels_in,
        num_channels_out,
        shape_kernel_conv,
        kwargs_conv=None,
        name_layer_norm=None,
        kwargs_norm=None,
        name_layer_act=None,
        kwargs_act=None,
        name_layer_pool=None,
        kwargs_pool=None,
        prob_dropout=None,
        kwargs_dropout=None,
    ):
        self.kwargs_act = kwargs_act or {}
        self.kwargs_dropout = kwargs_dropout or {}
        self.kwargs_conv = kwargs_conv or {}
        self.kwargs_norm = kwargs_norm or {}
        self.kwargs_pool = kwargs_pool or {}
        self.name_layer_act = name_layer_act
        self.name_layer_norm = name_layer_norm
        self.name_layer_pool = name_layer_pool
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.shape_kernel_conv = shape_kernel_conv
        self.prob_dropout = prob_dropout

        self._init()

    def _init(self):
        modules = [torch.nn.Conv2d(self.num_channels_in, self.num_channels_out, self.shape_kernel_conv, **self.kwargs_conv)]
        if self.name_layer_norm is not None:
            class_module = getattr(torch.nn, self.name_layer_norm)
            if self.name_layer_norm in ["BatchNorm2d", "InstanceNorm2d"]:
                self.kwargs_norm["num_features"] = self.num_channels_out
            modules += [class_module(**self.kwargs_norm)]
        if self.name_layer_act is not None:
            class_module = getattr(torch.nn, self.name_layer_act)
            modules += [class_module(**self.kwargs_act)]
        if self.name_layer_pool is not None:
            class_module = getattr(torch.nn, self.name_layer_pool)
            modules += [class_module(**self.kwargs_pool)]
        if self.prob_dropout is not None:
            modules += [torch.nn.Dropout(self.prob_dropout, **self.kwargs_dropout)]

        super().__init__(*modules)
