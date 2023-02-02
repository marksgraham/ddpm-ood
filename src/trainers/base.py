import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.get_train_and_val_dataloader import get_training_data_loader


class BaseTrainer:
    def __init__(self, args):
        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # initialise DDP if run was launched with torchrun
        if "LOCAL_RANK" in os.environ:
            print("Setting up DDP.")
            self.ddp = True
            # disable logging for processes except 0 on every node
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f

            # initialize the distributed training process, every GPU runs in a process
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.ddp = False
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        # set up dataloaders
        self.train_loader, self.val_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.training_ids,
            validation_ids=args.validation_ids,
            augmentation=bool(args.augmentation),
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            is_grayscale=bool(args.is_grayscale),
        )
        # set up model
        self.model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1 if args.is_grayscale else 3,
            out_channels=1 if args.is_grayscale else 3,
            num_channels=(128, 256, 256),
            attention_levels=(False, False, True),
            num_res_blocks=1,
            num_head_channels=256,
            with_conditioning=False,
        ).to(self.device)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
        )
        self.inferer = DiffusionInferer(self.scheduler)

        self.scaler = GradScaler()

        # set up optmizer, loss, checkpoints
        self.run_dir = Path(args.output_dir) / args.model_name
        checkpoint_path = self.run_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            print(
                f"Resuming training using checkpoint {checkpoint_path} at epoch {self.start_epoch}"
            )
        else:
            self.start_epoch = 0
            self.best_loss = 1000
            self.global_step = 0
        self.num_epochs = args.n_epochs

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
        if checkpoint_path.exists():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # wrap the model with DistributedDataParallel module
        if self.ddp:
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))

    def train(self, args):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                self.save_checkpoint(
                    self.run_dir / "checkpoint.pth",
                    epoch,
                    save_message=f"Saving checkpoint for model with loss {self.best_loss}",
                )

            if args.checkpoint_every != 0 and (epoch + 1) % args.checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir / f"checkpoint_{epoch+1}.pth",
                    epoch,
                    save_message=f"Saving checkpoint at epoch {epoch+1}",
                )

            if (epoch + 1) % args.eval_freq == 0:
                self.model.eval()
                self.val_epoch(epoch)
        print("Training completed.")
        if self.ddp:
            dist.destroy_process_group()

    def train_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=70,
            position=0,
            leave=True,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        epoch_step = 0
        self.model.train()
        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                # noise images
                noise = torch.randn_like(images).to(self.device)

                timesteps = torch.randint(
                    0,
                    self.inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()

                noise_prediction = self.inferer(
                    inputs=images, diffusion_model=self.model, noise=noise, timesteps=timesteps
                )
                loss = F.mse_loss(noise_prediction.float(), noise.float())
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()
            self.global_step += images.shape[0]
            epoch_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / epoch_step,
                }
            )

            self.logger_train.add_scalar(
                tag="loss", scalar_value=loss.item(), global_step=self.global_step
            )
            if self.quick_test:
                break
        epoch_loss = epoch_loss / epoch_step
        return epoch_loss

    def val_epoch(self, epoch):
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                ncols=70,
                position=0,
                leave=True,
                desc="Validation",
            )
            epoch_loss = 0
            epoch_step = 0
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    # noise images + segs
                    noise = torch.randn_like(images).to(self.device)

                    timesteps = torch.randint(
                        0,
                        self.inferer.scheduler.num_train_timesteps,
                        (images.shape[0],),
                        device=images.device,
                    ).long()

                    noise_prediction = self.model(x=images, timesteps=timesteps)
                    loss = F.mse_loss(noise_prediction.float(), noise.float())
                epoch_loss += loss.item()
                epoch_step += images.shape[0]
                progress_bar.set_postfix(
                    {
                        "loss": epoch_loss / epoch_step,
                    }
                )
            epoch_loss = epoch_loss / epoch_step
            self.logger_val.add_scalar(
                tag="loss", scalar_value=epoch_loss, global_step=self.global_step
            )

        # get some samples
        noise = torch.randn((8, images.shape[1], images.shape[2], images.shape[3])).to(self.device)
        samples = self.inferer.sample(
            input_noise=noise, diffusion_model=self.model, scheduler=self.scheduler, verbose=False
        )
        fig, ax = plt.subplots(2, 4)
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(np.transpose(samples[i, ...].cpu().numpy(), (1, 2, 0)), cmap="gray")
            plt.axis("off")
        self.logger_val.add_figure(tag="samples", figure=fig, global_step=self.global_step)

    def save_checkpoint(self, path, epoch, save_message=None):
        if self.ddp and dist.get_rank() == 0:
            # if DDP save a state dict that can be loaded by non-parallel models
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        if not self.ddp:
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
