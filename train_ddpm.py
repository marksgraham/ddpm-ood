import argparse
import ast

from src.trainers import DDPMTrainer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument(
        "--spatial_dimension", default=2, type=int, help="Dimension of images: 2d or 3d."
    )
    parser.add_argument("--image_size", default=None, help="Resize images.")
    parser.add_argument(
        "--image_roi",
        default=None,
        help="Specify central ROI crop of inputs, as a tuple, with -1 to not crop a dimension.",
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--latent_pad",
        default=None,
        help="Specify padding to apply to a latent, sometimes necessary to allow the DDPM U-net to work. Supply as a "
        "tuple following the 'pad' argument of torch.nn.functional.pad",
        type=ast.literal_eval,
    )
    # model params
    parser.add_argument(
        "--vqvae_checkpoint",
        default=None,
        help="Path to a VQ-VAE model checkpoint, if you wish to train an LDM.",
    )
    parser.add_argument(
        "--prediction_type",
        default="epsilon",
        help="Scheduler prediction type to use: 'epsilon, sample, or v_prediction.",
    )
    parser.add_argument(
        "--model_type",
        default="small",
        help="Small or big model.",
    )
    parser.add_argument(
        "--beta_schedule",
        default="linear",
        help="Linear or scaled linear",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=1e-4,
        help="Beta start.",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=2e-2,
        help="Beta end.",
    )
    parser.add_argument(
        "--b_scale",
        type=float,
        default=1,
        help="Scale the data by a factor b before noising.",
    )
    parser.add_argument(
        "--snr_shift",
        type=float,
        default=1,
        help="Shift the SNR of the noise scheduler by a factor to account for it increasing at higher resolution.",
    )
    parser.add_argument(
        "--simplex_noise",
        type=int,
        default=0,
        help="Use simplex instead of Gaussian noise.",
    )
    # training param
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Number of epochs to between evaluations.",
    )
    parser.add_argument(
        "--augmentation",
        type=int,
        default=1,
        help="Use of augmentation, 1 (True) or 0 (False).",
    )

    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument(
        "--cache_data",
        type=int,
        default=1,
        help="Whether or not to cache data in dataloaders.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save a checkpoint every checkpoint_every epochs.",
    )
    parser.add_argument(
        "--ddpm_checkpoint_epoch",
        default=None,
        help="If resuming, the epoch number for a specific checkpoint to resume from. If not specified, defaults to the best checkpoint.",
    )
    parser.add_argument("--is_grayscale", type=int, default=0, help="Is data grayscale.")
    parser.add_argument(
        "--quick_test",
        default=0,
        type=int,
        help="If True, runs through a single batch of the train and eval loop.",
    )
    args = parser.parse_args()
    return args


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_ddpm.py --args
if __name__ == "__main__":
    args = parse_args()
    trainer = DDPMTrainer(args)
    trainer.train(args)
