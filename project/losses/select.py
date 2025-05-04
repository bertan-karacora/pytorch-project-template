import torch

import self_supervised_learning_of_depth_and_motion.libs.utils_import as utils_import


class Select(torch.nn.Module):
    def __init__(self, key, name_module, kwargs_module):
        super().__init__()

        self.key = key
        self.kwargs_module = kwargs_module
        self.module = None
        self.name_module = name_module

        class_module = utils_import.import_criterion(self.name_module)
        self.module = class_module(**self.kwargs_module)

    def forward(self, input, target):
        output = self.module(input[self.key], target)
        return output
