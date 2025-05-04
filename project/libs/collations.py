import torch

import self_supervised_learning_of_depth_and_motion.libs.utils_data as utils_data


def create_collation_default(dict_transform):
    transform = utils_data.create_transform(dict_transform)

    def collate_fn(batch):
        return transform(*torch.utils.data.default_collate(batch))

    return collate_fn
