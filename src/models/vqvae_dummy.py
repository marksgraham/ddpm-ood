from torch import nn
import torch
from typing import Dict, List, Sequence, Union, Tuple


class DummyVQVAE(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        return [images]

    def quantize(self, images: torch.Tensor) -> List[torch.Tensor]:
        return [images]

    def decode(self, images: torch.Tensor) -> List[torch.Tensor]:
        return [images]

    def index_quantize(self, images: torch.Tensor) -> List[torch.Tensor]:
        return [images]

    def get_ldm_inputs(self, images: torch.Tensor) -> List[torch.Tensor]:
        return images



    def reconstruct_ldm_outputs(self, images: torch.Tensor) -> List[torch.Tensor]:
        return  images

    def pad_ldm_inputs(self, images: torch.Tensor) -> List[torch.Tensor]:
        return images

    def crop_ldm_inputs(self, images: torch.Tensor) -> List[torch.Tensor]:
        return images

    def decode_samples(self, embedding_indices: List[torch.Tensor]) -> torch.Tensor:
        samples_codes = self.quantizer[0].embed(embedding_indices[0].squeeze(dim=1))
        samples_images = self.decode([samples_codes])

        return samples_images

    def forward(self, images: torch.Tensor, get_ldm_inputs=False) -> Dict[str, List[torch.Tensor]]:
        # if statement allows the use of forward() in DataParallel mode to get ldm inputs
        if get_ldm_inputs:
            return self.get_ldm_inputs(images)
        else:
            return {
                "reconstruction": [images],
                "quantization_losses": 0,
            }
