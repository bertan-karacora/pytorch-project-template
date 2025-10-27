import logging

import torch

import project.libs.factory as factory


_LOGGER = logging.getLogger(__name__)


def create_collation_default(dict_transform):
    transform = factory.create_transform(dict_transform)

    def collate(batch):
        return transform(*torch.utils.data.default_collate(batch))

    return collate
