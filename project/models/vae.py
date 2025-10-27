import logging

import torch
import torchvision.transforms.v2 as tv_transforms

import project.libs.utils_import as utils_import


_LOGGER = logging.getLogger(__name__)


class VAEGaussian(torch.nn.Module):
    def __init__(
        self,
        num_channels_out_encode,
        num_channels_latent,
        shape_input_decoder,
        name_encoder,
        kwargs_encoder,
        name_decoder,
        kwargs_decoder,
        use_bias=True,
        use_normalize=False,
        kwargs_normalize=None,
    ):
        super().__init__()

        self.decoder = None
        self.encoder = None
        self.kwargs_decoder = kwargs_decoder or {}
        self.kwargs_encoder = kwargs_encoder or {}
        self.kwargs_normalize = kwargs_normalize or {}
        self.linear_encode_mean = None
        self.linear_encode_log_var = None
        self.linear_decode = None
        self.name_decoder = name_decoder
        self.name_encoder = name_encoder
        self.num_channels_out_encode = num_channels_out_encode
        self.num_channels_latent = num_channels_latent
        self.shape_input_decoder = shape_input_decoder
        self.use_bias = use_bias
        self.use_normalize = use_normalize

        self._init()

    def _init(self):
        self.linear_mean = torch.nn.Linear(self.num_channels_out_encode, self.num_channels_latent, bias=self.use_bias)
        self.linear_log_var = torch.nn.Linear(self.num_channels_out_encode, self.num_channels_latent, bias=self.use_bias)
        self.linear_decode = torch.nn.Linear(self.num_channels_latent, self.num_channels_out_encode, bias=self.use_bias)

        class_model = utils_import.import_model(self.name_encoder)
        self.encoder = class_model(**self.kwargs_encoder)

        class_model = utils_import.import_model(self.name_decoder)
        self.decoder = class_model(**self.kwargs_decoder)

    def encode(self, input):
        output = self.encoder(input)
        output = torch.flatten(output, start_dim=1)

        mean = self.linear_mean(output)
        log_var = self.linear_log_var(output)
        return mean, log_var

    def decode(self, input):
        output = self.linear_decode(input)
        output = output.view(-1, *self.shape_input_decoder)
        output = self.decoder(output)
        output = torch.sigmoid(output)

        if self.use_normalize:
            output = tv_transforms.functional.normalize(output, **self.kwargs_normalize)

        return output

    def reparameterize(self, mean, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        code_latent = mean + std * torch.randn_like(std)
        return code_latent

    def forward(self, input):
        mean, log_var = self.encode(input)
        code_latent = self.reparameterize(mean, log_var)
        output = self.decode(code_latent)

        output = {
            "prediction": output,
            "mean": mean,
            "log_var": log_var,
        }
        return output

    @torch.no_grad()
    def generate(self, num_samples=16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        code_latent = torch.randn(num_samples, self.num_channels_latent, device=device)
        samples = self.decode(code_latent)
        return samples
