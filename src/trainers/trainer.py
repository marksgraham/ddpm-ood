import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.get_train_and_val_dataloader import get_training_data_loader

from .base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

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
            augmentation=bool(args.augmentation),
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            is_grayscale=bool(args.is_grayscale),
        )

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
                    device=self.device,
                ).long()

                noisy_image = self.scheduler.add_noise(
                    original_samples=images * self.b_scale, noise=noise, timesteps=timesteps
                )

                noise_prediction = self.model(x=noisy_image, timesteps=timesteps)
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
            global_val_step = self.global_step
            val_steps = 0
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
                    noisy_image = self.scheduler.add_noise(
                        original_samples=images * self.b_scale, noise=noise, timesteps=timesteps
                    )
                    noise_prediction = self.model(x=noisy_image, timesteps=timesteps)
                    loss = F.mse_loss(noise_prediction.float(), noise.float())
                self.logger_val.add_scalar(
                    tag="loss", scalar_value=loss.item(), global_step=global_val_step
                )
                epoch_loss += loss.item()
                val_steps += images.shape[0]
                global_val_step += images.shape[0]
                progress_bar.set_postfix(
                    {
                        "loss": epoch_loss / val_steps,
                    }
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
