import logging

import torch

import project.libs.utils_import as utils_import


_LOGGER = logging.getLogger(__name__)


class RNNClassifier(torch.nn.Module):
    def __init__(
        self,
        name_encoder="CNN2dEncoder",
        kwargs_encoder=None,
        name_rnn="rnn",
        kwargs_rnn=None,
        name_head="MLP",
        kwargs_head=None,
    ):
        super().__init__()

        self.encoder = None
        self.head = None
        self.kwargs_encoder = kwargs_encoder or {}
        self.kwargs_head = kwargs_head or {}
        self.kwargs_rnn = kwargs_rnn or {}
        self.rnn = None
        self.name_encoder = name_encoder
        self.name_head = name_head
        self.name_rnn = name_rnn

        self._init()

    def _init(self):
        class_model = utils_import.import_model(self.name_encoder)
        self.encoder = class_model(**self.kwargs_encoder)

        class_model = utils_import.import_model(self.name_rnn)
        self.rnn = class_model(**self.kwargs_rnn)

        class_model = utils_import.import_model(self.name_head)
        self.head = class_model(**self.kwargs_head)

    def forward(self, input):
        output_frames = []
        for i in range(input.shape[1]):
            output_frames += [self.encoder(input[:, i, ...])]
        output = torch.stack(output_frames, dim=1)
        output = self.rnn(output)
        output = self.head(output)
        return output
