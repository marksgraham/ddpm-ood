import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from src.models.ddpm_2d import DDPM
from src.models.vqvae_2d import BaselineVQVAE2D
from src.models.vqvae_dummy import DummyVQVAE
from src.training_and_testing.training_functions import train_ldm
from src.training_and_testing.util import get_training_data_loader

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--out_ids", help="List of location of file with outlier ids.")
    parser.add_argument(
        "--config_vqvae_file",
        default="None",
        help="Location of VQ-VAE config. None if not training a latent diffusion model.",
    )
    parser.add_argument("--config_diffusion_file", help="Location of config.")
    parser.add_argument("--vqvae_checkpoint", help="VQVAE checkpoint path.")
    # training param
    parser.add_argument("--batch_size", type=int, default=180, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Number of epochs to betweeen evaluations.",
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
    parser.add_argument("--is_grayscale", type=int, default=0, help="Is data grayscale.")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()
    run_dir = Path(args.output_dir) / args.model_name
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True, parents=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # initialise DDP if run was launched with torchrun
    if "LOCAL_RANK" in os.environ:
        print("Setting up DDP.")
        ddp = True
        # disable logging for processes except 0 on every node
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank != 0:
            f = open(os.devnull, "w")
            sys.stdout = sys.stderr = f

        # initialize the distributed training process, every GPU runs in a process
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        ddp = False
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    print("Getting data...")
    train_loader, val_loader = get_training_data_loader(
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        augmentation=bool(args.augmentation),
        num_workers=args.num_workers,
        cache_data=bool(args.cache_data),
        is_grayscale=bool(args.is_grayscale),
    )

    # Load VQVAE to produce the encoded samples
    if args.config_vqvae_file != "None":
        config_vqvae = OmegaConf.load(args.config_vqvae_file)
        vqvae = BaselineVQVAE2D(**config_vqvae["stage1"])
        if os.environ["HOME"] == "/root":
            checkpoint = torch.load(args.vqvae_checkpoint)
        else:
            checkpoint = torch.load(args.vqvae_checkpoint)
        print(f"Loaded VQVAE checkpoing {args.vqvae_checkpoint}")
        vqvae.load_state_dict(checkpoint["network"])
        vqvae.eval()
    else:
        vqvae = DummyVQVAE()

    # Create model
    print("Creating model...")
    config_ldm = OmegaConf.load(args.config_diffusion_file)
    diffusion = DDPM(**config_ldm["ldm"].get("params", dict()))

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    vqvae = vqvae.to(device)
    diffusion = diffusion.to(device)

    raw_diffusion = diffusion.module if hasattr(diffusion, "module") else diffusion
    optimizer = optim.Adam(diffusion.parameters(), lr=config_ldm["ldm"]["base_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    best_nll = float("inf")
    start_epoch = 0
    if resume:
        print("Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        diffusion.load_state_dict(checkpoint["diffusion"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        best_nll = checkpoint["best_nll"]
        if "t_sampler_history" in checkpoint.keys():
            raw_diffusion.t_sampler._loss_history = checkpoint["t_sampler_history"]
            raw_diffusion.t_sampler._loss_counts = checkpoint["t_sampler_loss_counts"]

    else:
        print("No checkpoint found.")

    if ddp:
        if args.config_vqvae_file != "None":
            vqvae = DistributedDataParallel(vqvae, device_ids=[device])
        diffusion = DistributedDataParallel(diffusion, device_ids=[device])

    # Train model
    print("Starting Training")
    val_loss = train_ldm(
        model=diffusion,
        vqvae=vqvae,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        checkpoint_every=args.checkpoint_every,
        best_nll=best_nll,
        ddp=ddp,
    )


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train.py --args
if __name__ == "__main__":
    args = parse_args()
    main(args)
