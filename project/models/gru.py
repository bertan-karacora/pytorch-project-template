import torch

import project.libs.utils_import as utils_import


class GRU(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out, num_cells=1, name_func_init="zeros_", use_bias=True):
        super().__init__()

        self.cells = None
        self.name_func_init = name_func_init
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out
        self.num_cells = num_cells
        self.use_bias = use_bias

        self._init()

    def _init(self):
        cells = []
        for i in range(self.num_cells):
            cells += [
                torch.nn.GRUCell(
                    input_size=self.num_channels_in if i == 0 else self.num_channels_out,
                    hidden_size=self.num_channels_out,
                    bias=self.use_bias,
                )
            ]
        self.cells = torch.nn.ModuleList(cells)

    def forward(self, input):
        states_hidden = self._init_states(input)

        outputs_cells = []
        for s in range(input.shape[1]):
            input_cell = input[:, s, ...]
            for c, gru_cell in enumerate(self.cells):
                states_hidden[c] = gru_cell(input_cell, states_hidden[c])
                input_cell = states_hidden[c]
            outputs_cells += [states_hidden[-1]]
        outputs_cells = torch.stack(outputs_cells, dim=1)

        return outputs_cells[:, -1, ...]

    def _init_states(self, input):
        states_hidden = [torch.empty(input.shape[0], self.num_channels_out, dtype=input.dtype, layout=input.layout, device=input.device) for _ in range(self.num_cells)]

        func = getattr(torch.nn.init, self.name_func_init)
        for c in range(self.num_cells):
            func(states_hidden[c])

        return states_hidden


class GRUClassifier(torch.nn.Module):
    def __init__(
        self,
        name_encoder="CNN2dEncoder",
        kwargs_encoder=None,
        name_gru="GRU",
        kwargs_gru=None,
        name_head="MLP",
        kwargs_head=None,
    ):
        super().__init__()

        self.encoder = None
        self.head = None
        self.kwargs_encoder = kwargs_encoder or {}
        self.kwargs_head = kwargs_head or {}
        self.kwargs_gru = kwargs_gru or {}
        self.gru = None
        self.name_encoder = name_encoder
        self.name_head = name_head
        self.name_gru = name_gru

        self._init()

    def _init(self):
        class_model = utils_import.import_model(self.name_encoder)
        self.encoder = class_model(**self.kwargs_encoder)

        class_model = utils_import.import_model(self.name_gru)
        self.gru = class_model(**self.kwargs_gru)

        class_model = utils_import.import_model(self.name_head)
        self.head = class_model(**self.kwargs_head)

    def forward(self, input):
        output_frames = []
        for i in range(input.shape[1]):
            output_frames += [self.encoder(input[:, i, ...])]
        output = torch.stack(output_frames, dim=1)
        output = self.gru(output)
        output = self.head(output)
        return output
