import torch


class PassthroughVQVAE(torch.nn.Module):
    """
    This fake VQ-VAE just returns inputs.
    """

    def __init__(self):
        super().__init__()
        self.latent_channels = 1

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode_stage_2_outputs(self, x: torch.Tensor) -> torch.Tensor:
        return x
