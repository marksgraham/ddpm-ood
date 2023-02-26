from typing import Dict, Tuple

import torch
from lpips import LPIPS


# It is a torch.nn.Module to be able to move the network used for the perceptual loss to the desired compute devices
class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss based on the lpips library. The 3D implementation is based on a 2.5D approach where we batchify
    every spatial dimension one after another so we obtain better spatial consistency. There is also a pixel
    component as well.

    Args:
        dimensions (int): Dimensions: number of spatial dimensions.
        include_pixel_loss (bool): If the loss includes the pixel component as well
        is_fake_3d (bool): Whether we use 2.5D approach for a 3D perceptual loss
        drop_ratio (float): How many, as a ratio, slices we drop in the 2.5D approach
        lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS constructor.
            Defaults to: { 'pretrained': True, 'net': 'alex', 'version': '0.1', 'lpips': True, 'spatial': False,
            'pnet_rand': False,  'pnet_tune': False, 'use_dropout': True, 'model_path': None, 'eval_mode': True,
            'verbose': True}
        lpips_normalize (bool): Whether or not the input needs to be renormalised from [0,1] to [-1,1]

    Attributes:
        self.dimensions (int): Number of spatial dimensions.
        self.include_pixel_loss (bool): If the loss includes the pixel component as well
                self.lpips_kwargs (Dict): Dictionary containing key words arguments that will be passed to the LPIPS
            constructor function call.
        self.fake_3D_views (List[Tuple[Tuple[int,int,int,int,int],Tuple[int,int,int]]]): List of pairs for the 2.5D
            approach. The first element in every tuple is the required permutation for an axis and the second one
            hold the indices of the input image that dictate the shape of the bachified tensor.
        self.keep_ratio (float): Ratio of how many elements of the every 2.5D view we are using to calculate the
            loss. This allows for a memory & iteration speed vs information flow compromise.
        self.lpips_normalize (bool): Whether or not we renormalize from [0,1] to [-1,1]
        self.perceptual_function (Callable): Function that calculates the perceptual loss. For 2D and 2.5D is based
            LPIPS. 3D is not implemented yet.
        self.perceptual_factor (float): Scaling factor of the perceptual component of loss
        self.summaries (Dict): Dictionary containing scalar summaries to be logged in TensorBoard

    References:
        [1] Zhang, R., Isola, P., Efros, A.A., Shechtman, E. and Wang, O., 2018.
        The unreasonable effectiveness of deep features as a perceptual metric.
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 586-595).
    """

    def __init__(
        self,
        dimensions: int,
        include_pixel_loss: bool = True,
        is_fake_3d: bool = True,
        drop_ratio: float = 0.0,
        fake_3d_axis: Tuple[int, ...] = (2, 3, 4),
        lpips_kwargs: Dict = None,
        lpips_normalize: bool = True,
        spatial: bool = False,
    ):
        super(PerceptualLoss, self).__init__()

        if not (dimensions in [2, 3]):
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if dimensions == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        self.dimensions = dimensions
        self.include_pixel_loss = include_pixel_loss
        self.lpips_kwargs = (
            {
                "pretrained": True,
                "net": "alex",
                "version": "0.1",
                "lpips": True,
                "spatial": spatial,
                "pnet_rand": False,
                "pnet_tune": False,
                "use_dropout": True,
                "model_path": None,
                "eval_mode": True,
                "verbose": False,
            }
            if lpips_kwargs is None
            else lpips_kwargs
        )
        # Here we store the permutations of the 5D tensor where we merge different axis into the batch dimension
        # and use the rest as spatial dimensions, we allow
        self.fake_3D_views = (
            (
                []
                + ([((0, 2, 1, 3, 4), (1, 3, 4))] if 2 in fake_3d_axis else [])
                + ([((0, 3, 1, 2, 4), (1, 2, 4))] if 3 in fake_3d_axis else [])
                + ([((0, 4, 1, 2, 3), (1, 2, 3))] if 4 in fake_3d_axis else [])
            )
            if is_fake_3d
            else None
        )
        # In case of being memory constraint for the 2.5D approach it allows to randomly drop some slices
        self.keep_ratio = 1 - drop_ratio
        self.lpips_normalize = lpips_normalize
        self.perceptual_function = (
            LPIPS(**self.lpips_kwargs) if self.dimensions == 2 or is_fake_3d else None
        )
        self.perceptual_factor = 1

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Unpacking elements
        y = y.float()
        y_pred = y_pred.float()

        if self.dimensions == 3 and self.fake_3D_views:
            loss = torch.zeros(())

            for idx, fake_views in enumerate(self.fake_3D_views):
                loss = (
                    self._calculate_fake_3d_loss(
                        y=y,
                        y_pred=y_pred,
                        permute_dims=fake_views[0],
                        view_dims=fake_views[1],
                    )
                    * self.perceptual_factor
                )
        else:
            loss = (
                self.perceptual_function.forward(y, y_pred, normalize=self.lpips_normalize)
                * self.perceptual_factor
            )

        return loss

    def _calculate_fake_3d_loss(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        permute_dims: Tuple[int, int, int, int, int],
        view_dims: Tuple[int, int, int],
    ):
        """
        Calculating perceptual loss after one spatial axis is batchified according to permute dims and
        we drop random slices as per self.keep_ratio.

        Args:
            y (torch.Tensor): Ground truth images
            y_pred (torch.Tensor): Predictions
            permute_dims (Tuple[int,int,int,int,int]): The order in which the permutation happens where the first
                to newly permuted dimensions are going to become the batch dimension
            view_dims (Tuple[int,int,int]): The channel dimension and two spatial dimensions that are being kept
                after the permutation.

        Returns:
            torch.Tensor: perceptual loss value on the given axis
        """
        # Reshaping the ground truth and prediction to be considered 2D
        y_slices = (
            y.permute(*permute_dims)
            .contiguous()
            .view(-1, y.shape[view_dims[0]], y.shape[view_dims[1]], y.shape[view_dims[2]])
        )

        y_pred_slices = (
            y_pred.permute(*permute_dims)
            .contiguous()
            .view(
                -1,
                y_pred.shape[view_dims[0]],
                y_pred.shape[view_dims[1]],
                y_pred.shape[view_dims[2]],
            )
        )

        # Subsampling in case we are memory constrained
        indices = torch.randperm(y_pred_slices.shape[0], device=y_pred_slices.device)[
            : int(y_pred_slices.shape[0] * self.keep_ratio)
        ]

        y_pred_slices = y_pred_slices.as_tensor()[indices]
        y_slices = y_slices.as_tensor()[indices]

        # Calculating the 2.5D perceptual loss
        p_loss = torch.mean(
            self.perceptual_function.forward(
                y_slices, y_pred_slices, normalize=self.lpips_normalize
            )
        )

        return p_loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_perceptual_factor(self) -> float:
        return self.perceptual_factor

    def set_perceptual_factor(self, perceptual_factor: float) -> float:
        self.perceptual_factor = perceptual_factor

        return self.get_perceptual_factor()
