import torch
import torch.nn as nn
import torch.distributed.distributed_c10d as dist

from typing import Dict, List, Sequence, Union, Tuple

from torch.nn import functional as F

from .vqvae_base import VQVAEBase

# Isolated baseline implementation provided by @warvito and should be used as minimum performance
# for any further improvement. This implementation is 3D only. It was extracted from his codebase
# at commit https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/
# Based on following files
# - Network - https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/models/vqvae.py
# - Quantizer - https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/layers/quantizer.py
# - Default values - https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/runai/submit_train_vqgan.sh
# Modifications were done to make it compliant with the VQVAE interface


# This is an implementation trick to work around a DDP and AMP where FP16 were encountered during high batch size
# training. Encapsulating the forward function in a method was tried but it resulted in AMP mishandling the output
# type.
class Quantizer_impl(nn.Module):
    def __init__(self, n_embed, embed_dim, eps):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.eps = eps

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.requires_grad = False
        self.weight = self.embedding.weight

        self.register_buffer("N", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.weight.data.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self, x: torch.Tensor, decay: float, commitment_cost: float
    ) -> List[torch.Tensor]:
        b, c, h, w = x.shape
        x = x.float()

        # convert inputs from BCHW -> BHWC and flatten input
        flat_inputs = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embed_dim)
        # Calculate distances
        distances = (
            (flat_inputs ** 2).sum(dim=1, keepdim=True)
            - 2 * torch.mm(flat_inputs, self.weight.t())
            + (self.weight ** 2).sum(dim=1, keepdim=True).t()
        )

        # Encoding
        embed_idx = torch.max(-distances, dim=1)[1]
        embed_onehot = F.one_hot(embed_idx, self.n_embed).type_as(flat_inputs)

        # Quantize and unflatten
        embed_idx = embed_idx.view(b, h, w)

        # Embed
        quantized = self.embedding(embed_idx).permute(0, 3, 1, 2).contiguous()

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                encodings_sum = embed_onehot.sum(0)
                dw = torch.mm(embed_onehot.t(), flat_inputs)
                if dist.is_initialized():
                    dist.all_reduce(tensor=encodings_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tensor=dw, op=dist.ReduceOp.SUM)

                # Laplace smoothing of the cluster size
                self.N.data.mul_(decay).add_(torch.mul(encodings_sum, 1 - decay))
                self.embed_avg.data.mul_(decay).add_(torch.mul(dw, 1 - decay))

                n = self.N.sum()
                W = (self.N + self.eps) / (n + self.n_embed * self.eps) * n
                self.weight.data.copy_(self.embed_avg / W.unsqueeze(1))

        latent_loss = commitment_cost * F.mse_loss(quantized.detach(), x)

        # Stop optimization from accessing the embedding
        quantized_st = (quantized - x).detach() + x

        return quantized_st, latent_loss, embed_idx

    @torch.cuda.amp.autocast(enabled=False)
    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(embedding_indices).permute(0, 3, 1, 2).contiguous()


class Quantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.impl = Quantizer_impl(n_embed, embed_dim, eps)

        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.perplexity_code: torch.Tensor = torch.rand(1)

    def forward(self, x):
        quantized_st, latent_loss, embed_idx = self.impl(
            x, self.decay, self.commitment_cost
        )

        avg_probs = (
            lambda e: torch.histc(e.float(), bins=self.n_embed, max=self.n_embed)
            .float()
            .div(e.numel())
        )

        perplexity = lambda avg_probs: torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        self.perplexity_code = perplexity(avg_probs(embed_idx))

        return quantized_st, latent_loss

    def get_ema_decay(self) -> float:
        return self.decay

    def set_ema_decay(self, decay: float) -> float:
        self.decay = decay

        return self.get_ema_decay()

    def get_commitment_cost(self) -> float:
        return self.commitment_cost

    def set_commitment_cost(self, commitment_cost) -> float:
        self.commitment_cost = commitment_cost

        return self.get_commitment_cost()

    def get_perplexity(self) -> torch.Tensor:
        return self.perplexity_code

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.impl.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        return self.impl(encodings, self.decay, self.commitment_cost)


class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, n_res_channels, p_dropout):
        super().__init__(
            nn.Conv2d(n_channels, n_res_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout2d(p_dropout),
            nn.Conv2d(n_res_channels, n_channels, kernel_size=1),
        )

    def forward(self, x):
        return F.relu(x + super().forward(x), True)


class BaselineVQVAE2D(VQVAEBase, nn.Module):
    def __init__(
        self,
        n_levels=3,
        n_embed=256,
        embed_dim=256,
        n_channels=144,
        n_res_channels=144,
        n_res_layers=3,
        p_dropout=0.0,
        commitment_cost=0.25,
        vq_decay=0.5,
        n_input_channels=1,
        padding=(0, 0),
        encoding_type = 'quantised'

    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.n_res_channels = n_res_channels
        self.n_res_layers = n_res_layers
        self.p_dropout = p_dropout
        self.commitment_cost = commitment_cost
        self.vq_decay = vq_decay
        self.padding = padding
        self.n_input_channels = n_input_channels
        self.encoding_type = encoding_type

        self.encoder = self.construct_encoder()
        self.quantizer = self.construct_quantizer()
        self.decoder = self.construct_decoder()

        self.n_embed = n_embed

    def construct_encoder(self) -> nn.ModuleList:
        modules = []

        for i in range(self.n_levels):
            modules.append(
                nn.Conv2d(
                    self.n_input_channels if i == 0 else self.n_channels // 2,
                    self.n_channels // (1 if i == self.n_levels - 1 else 2),
                    4,
                    stride=2,
                    padding=1,
                )
            )
            modules.append(nn.ReLU())
            modules.append(
                nn.Sequential(
                    *[
                        ResidualLayer(
                            self.n_channels // (1 if i == self.n_levels - 1 else 2),
                            self.n_res_channels // (1 if i == self.n_levels - 1 else 2),
                            self.p_dropout,
                        )
                        for _ in range(self.n_res_layers)
                    ]
                )
            )

        modules.append(
            nn.Conv2d(self.n_channels, self.embed_dim, 3, stride=1, padding=1)
        )

        return nn.ModuleList([nn.Sequential(*modules)])

    def construct_quantizer(self) -> nn.ModuleList:
        quantizer = Quantizer(
            self.n_embed,
            self.embed_dim,
            commitment_cost=self.commitment_cost,
            decay=self.vq_decay,
        )
        return nn.ModuleList([quantizer])

    def construct_decoder(self) -> nn.ModuleList:
        modules = [nn.Conv2d(self.embed_dim, self.n_channels, 3, stride=1, padding=1)]

        for i in range(self.n_levels):
            modules.append(
                nn.Sequential(
                    *[
                        ResidualLayer(
                            self.n_channels // (1 if i == 0 else 2),
                            self.n_res_channels // (1 if i == 0 else 2),
                            self.p_dropout,
                        )
                        for _ in range(self.n_res_layers)
                    ]
                )
            )
            modules.append(
                nn.ConvTranspose2d(
                    self.n_channels // (1 if i == 0 else 2),
                    (self.n_input_channels if i == self.n_levels - 1 else self.n_channels // 2),
                    4,
                    stride=2,
                    padding=1,
                )
            )
            # We do not have an output activation
            if i != self.n_levels - 1:
                modules.append(nn.ReLU())

        return nn.ModuleList([nn.Sequential(*modules)])

    def get_ema_decay(self) -> Sequence[float]:
        return [self.quantizer[0].get_ema_decay()]

    def set_ema_decay(self, decay: Union[Sequence[float], float]) -> Sequence[float]:
        self.quantizer[0].set_ema_decay(decay[0] if isinstance(decay, list) else decay)

        return self.get_ema_decay()

    def get_commitment_cost(self) -> Sequence[float]:
        return [self.quantizer[0].get_commitment_cost()]

    def set_commitment_cost(
        self, commitment_factor: Union[Sequence[float], float]
    ) -> Sequence[float]:
        self.quantizer[0].set_commitment_cost(
            commitment_factor[0]
            if isinstance(commitment_factor, list)
            else commitment_factor
        )

        return self.get_commitment_cost()

    def get_perplexity(self) -> Sequence[float]:
        return [self.quantizer[0].get_perplexity()]

    def get_last_layer(self) -> nn.parameter.Parameter:
        return list(self.decoder.modules())[-1].weight

    def encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        return [self.encoder[0](images)]

    def quantize(
        self, encodings: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x, x_loss = self.quantizer[0](encodings[0])
        return [x], [x_loss]

    def decode(self, quantizations: List[torch.Tensor]) -> torch.Tensor:
        x = self.decoder[0](quantizations[0])
        return x

    def index_quantize(self, images: torch.Tensor) -> List[torch.Tensor]:
        encodings = self.encode(images)
        _, _, encoding_indices = self.quantizer[0].quantize(encodings[0])

        return [encoding_indices]

    def get_ldm_inputs(self, images: torch.Tensor) -> List[torch.Tensor]:
        if self.encoding_type == 'non_quantised':
            return self.encode(images)[0]
        elif self.encoding_type == 'indices':
            #return self.index_quantize(images)[0][:,None,...].float()
            indices = self.index_quantize(images)[0][:, None, ...].float()
            indices_norm = (indices/ (self.n_embed - 1))/0.5 -1
            return indices_norm
        elif self.encoding_type == 'quantised':
            encoding = self.encode(images)[0]
            quantizations, quantization_losses = self.quantize([encoding])
            return quantizations[0]
        elif self.encoding_type == 'downsampling':
            downsample_factor = pow(2,self.n_levels)
            downsampled = F.interpolate(images, scale_factor=1/downsample_factor)
            return downsampled

    def reconstruct_ldm_outputs(self, encodings: torch.Tensor) -> torch.Tensor:
        if self.encoding_type == 'non_quantised':
            quantizations, quantization_losses = self.quantize([encodings])
            reconstruction = self.decode(quantizations)
            return reconstruction
        elif self.encoding_type == 'indices':
            indices_unnormed = torch.round(((encodings + 1)*0.5) * (self.n_embed - 1)).int()
            indices_unnormed.clamp_(0, self.n_embed - 1)
            return self.decode_samples([indices_unnormed])
        elif self.encoding_type == 'quantised':
            reconstruction = self.decode([encodings])
            return reconstruction
        elif self.encoding_type == 'downsampling':
            downsample_factor = pow(2,self.n_levels)
            upsampled = F.interpolate(encodings, scale_factor=downsample_factor)
            return upsampled

    def pad_ldm_inputs(self, encodings: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(encodings, (0, self.padding[0],
                                                   0, self.padding[1]))

    def crop_ldm_inputs(self, encodings: torch.Tensor) -> torch.Tensor:
        padding = [-p if p!=0 else encodings.shape[i+2] for i,p in enumerate(self.padding)]
        return encodings[:, :,
               :padding[0],
               :padding[1]]

    def decode_samples(self, embedding_indices: List[torch.Tensor]) -> torch.Tensor:
        samples_codes = self.quantizer[0].embed(embedding_indices[0])
        samples_images = self.decode([samples_codes])

        return samples_images

    def forward(self, images: torch.Tensor, get_ldm_inputs=False) -> Dict[str, List[torch.Tensor]]:
        # if statement allows the use of forward() in DataParallel mode to get ldm inputs
        if get_ldm_inputs:
            return self.get_ldm_inputs(images)
        else:
            encodings = self.encode(images)
            quantizations, quantization_losses = self.quantize(encodings)
            reconstruction = self.decode(quantizations)

            return {
                "reconstruction": [reconstruction],
                "quantization_losses": quantization_losses,
            }
