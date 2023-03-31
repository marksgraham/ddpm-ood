import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.losses.spectral_loss import JukeboxLoss
from generative.networks.nets import VQVAE, PatchDiscriminator
from monai.networks.layers import Act
from torch.nn import L1Loss

# from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.get_train_and_val_dataloader import get_training_data_loader


class VQVAETrainer:
    def __init__(self, args):

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

        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # set up model
        self.spatial_dimension = args.spatial_dimension
        vqvae_args = {
            "spatial_dims": args.spatial_dimension,
            "in_channels": args.vqvae_in_channels,
            "out_channels": args.vqvae_out_channels,
            "num_res_layers": args.vqvae_num_res_layers,
            "downsample_parameters": args.vqvae_downsample_parameters,
            "upsample_parameters": args.vqvae_upsample_parameters,
            "num_channels": args.vqvae_num_channels,
            "num_res_channels": args.vqvae_num_res_channels,
            "num_embeddings": args.vqvae_num_embeddings,
            "embedding_dim": args.vqvae_embedding_dim,
            "decay": args.vqvae_decay,
            "commitment_cost": args.vqvae_commitment_cost,
            "epsilon": args.vqvae_epsilon,
            "dropout": args.vqvae_dropout,
            "ddp_sync": args.vqvae_ddp_sync,
        }
        self.model = VQVAE(**vqvae_args)
        self.model.to(self.device)
        print(f"{sum(p.numel() for p in self.model.parameters()):,} model parameters")

        self.discriminator = PatchDiscriminator(
            spatial_dims=args.spatial_dimension,
            num_layers_d=3,
            num_channels=64,
            in_channels=args.vqvae_in_channels,
            out_channels=args.vqvae_out_channels,
            kernel_size=4,
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )
        self.discriminator.to(self.device)

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=args.spatial_dimension, network_type="alex", is_fake_3d=True
        )
        self.perceptual_loss.to(self.device)
        self.jukebox_loss = JukeboxLoss(spatial_dims=args.spatial_dimension)
        self.jukebox_loss.to(self.device)
        self.optimizer_g = torch.optim.Adam(params=self.model.parameters(), lr=3e-4)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=5e-4)

        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.adv_weight = args.adversarial_weight
        self.perceptual_weight = 0.001
        self.adversarial_warmup = bool(args.adversarial_warmup)
        # set up optimizer, loss, checkpoints
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

        # save vqvae parameters
        self.run_dir.mkdir(exist_ok=True)
        with open(self.run_dir / "vqvae_config.json", "w") as f:
            json.dump(vqvae_args, f, indent=4)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
        if checkpoint_path.exists():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # wrap the model with DistributedDataParallel module
        if self.ddp:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            self.discriminator = DistributedDataParallel(
                self.discriminator,
                device_ids=[self.device],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.num_epochs = args.n_epochs
        self.train_loader, self.val_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.training_ids,
            validation_ids=args.validation_ids,
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            is_grayscale=bool(args.is_grayscale),
            spatial_dimension=args.spatial_dimension,
            image_size=int(args.image_size) if args.image_size else args.image_size,
            image_roi=args.image_roi,
        )

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
            ncols=110,
            position=0,
            leave=True,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        l1_loss = 0
        generator_epoch_loss = 0
        discriminator_epoch_loss = 0
        epoch_step = 0
        self.model.train()
        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            self.optimizer_g.zero_grad(set_to_none=True)

            # Generator part
            reconstruction, quantization_loss = self.model(images=images)
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

            recons_loss = self.l1_loss(reconstruction.float(), images.float())
            perceptual_loss = self.perceptual_loss(reconstruction.float(), images.float())
            jukebox_loss = self.jukebox_loss(reconstruction.float(), images.float())
            adversarial_loss = self.adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            if self.adversarial_warmup:
                adv_weight = self.adv_weight * min(epoch, 50) / 50
            else:
                adv_weight = self.adv_weight
            total_generator_loss = (
                recons_loss
                + quantization_loss
                + self.perceptual_weight * perceptual_loss
                + jukebox_loss
                + adv_weight * adversarial_loss
            )

            total_generator_loss.backward()
            self.optimizer_g.step()

            # Discriminator part
            self.optimizer_d.zero_grad(set_to_none=True)

            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = self.discriminator(images.contiguous().detach())[-1]
            loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            self.optimizer_d.step()

            l1_loss += recons_loss.item()
            generator_epoch_loss += total_generator_loss.item()
            discriminator_epoch_loss += discriminator_loss.item()
            epoch_step += images.shape[0]
            self.global_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "l1_loss": l1_loss / (epoch_step),
                    "generator_loss": generator_epoch_loss / (epoch_step),
                    "discriminator_loss": discriminator_epoch_loss / (epoch_step),
                }
            )

            self.logger_train.add_scalar(
                tag="l1_loss", scalar_value=recons_loss.item(), global_step=self.global_step
            )
            self.logger_train.add_scalar(
                tag="perceptual_loss",
                scalar_value=perceptual_loss.item(),
                global_step=self.global_step,
            )
            self.logger_train.add_scalar(
                tag="jukebox_loss", scalar_value=jukebox_loss.item(), global_step=self.global_step
            )
            self.logger_train.add_scalar(
                tag="adversarial_loss",
                scalar_value=adversarial_loss.item(),
                global_step=self.global_step,
            )
            self.logger_train.add_scalar(
                tag="generator_loss",
                scalar_value=total_generator_loss.item(),
                global_step=self.global_step,
            )
            self.logger_train.add_scalar(
                tag="discriminator_loss",
                scalar_value=discriminator_loss.item(),
                global_step=self.global_step,
            )
            if self.quick_test:
                break
        epoch_loss = generator_epoch_loss / epoch_step
        return epoch_loss

    def val_epoch(self, epoch):
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                ncols=110,
                position=0,
                leave=True,
                desc="Validation",
            )

            global_val_step = self.global_step
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                reconstruction, quantization_loss = self.model(images=images)
                logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

                recons_loss = self.l1_loss(reconstruction.float(), images.float())
                perceptual_loss = self.perceptual_loss(reconstruction.float(), images.float())
                jukebox_loss = self.jukebox_loss(reconstruction.float(), images.float())
                adversarial_loss = self.adv_loss(
                    logits_fake, target_is_real=True, for_discriminator=False
                )
                total_generator_loss = (
                    recons_loss
                    + quantization_loss
                    + self.perceptual_weight * perceptual_loss
                    + jukebox_loss
                    + self.adv_weight * adversarial_loss
                )

                self.logger_val.add_scalar(
                    tag="l1_loss", scalar_value=recons_loss.item(), global_step=global_val_step
                )
                self.logger_val.add_scalar(
                    tag="generator_loss",
                    scalar_value=total_generator_loss.item(),
                    global_step=global_val_step,
                )
                # self.logger_val.add_scalar(
                #     tag="discriminator_loss", scalar_value=disc_epoch_loss, global_step=global_val_step
                # )
                global_val_step += images.shape[0]

                # plot some recons
                if step == 0:
                    fig = plt.figure()
                    for i in range(2):
                        plt.subplot(2, 2, i * 2 + 1)
                        if self.spatial_dimension == 2:
                            sl = np.s_[i, 0, :, :]
                        else:
                            sl = np.s_[i, 0, :, :, images.shape[4] // 2]
                        plt.imshow(images[sl].cpu(), cmap="gray")
                        if i == 0:
                            plt.title("Image")
                        plt.subplot(2, 2, i * 2 + 2)
                        plt.imshow(reconstruction[sl].cpu(), cmap="gray")
                        if i == 0:
                            plt.title("Recon")
                    plt.show()
                    self.logger_val.add_figure(
                        tag="recons", figure=fig, global_step=self.global_step
                    )
