from collections import OrderedDict
from pathlib import PosixPath

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_ldm(
    model,
    vqvae,
    scheduler,
    inferer,
    start_epoch: int,
    best_loss: float,
    train_loader,
    val_loader,
    optimizer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: PosixPath,
    checkpoint_every: int,
    best_nll: float,
    ddp: bool = False,
):
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model
    quick_test = True
    if quick_test:
        print("WARNING: just running on one batch of train and val sets.")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
            model=model,
            vqvae=vqvae,
            scheduler=scheduler,
            inferer=inferer,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            quick_test=quick_test,
        )
        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ldm(
                model=model,
                vqvae=vqvae,
                scheduler=scheduler,
                inferer=inferer,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                quick_test=quick_test,
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            if ddp and dist.get_rank() == 0 or not ddp:
                checkpoint = {
                    "epoch": epoch + 1,
                    "diffusion": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_nll": best_nll,
                    "t_sampler_history": raw_model.t_sampler._loss_history,
                    "t_sampler_loss_counts": raw_model.t_sampler._loss_counts,
                }
                torch.save(checkpoint, str(run_dir / "checkpoint.pth"))
                if val_loss <= best_loss:
                    print(f"New best val loss {val_loss}")
                    best_loss = val_loss
                    torch.save(raw_model.state_dict(), str(run_dir / "best_model_val_loss.pth"))

    print("Training finished!")
    print("Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / "final_model.pth"))
    if ddp:
        dist.destroy_process_group()
    return val_loss


def train_epoch_ldm(
    model,
    vqvae,
    scheduler,
    inferer,
    loader,
    optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler,
    quick_test: bool,
):
    model.train()
    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        img = x["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                # e = raw_vqvae.get_ldm_inputs(img.to(device))
                inputs = vqvae(img.to(device), get_ldm_inputs=True)
            noise = torch.randn_like(inputs).to(device)
            timesteps = torch.randint(
                0,
                inferer.scheduler.num_train_timesteps,
                (inputs.shape[0],),
                device=device,
            ).long()
            noise_prediction = model(x=inputs, timesteps=timesteps)
            loss = F.mse_loss(noise_prediction.float(), noise.float())
            if quick_test:
                break
        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.5f}",
                "lr": f"{get_lr(optimizer):.6f}",
            }
        )


@torch.no_grad()
def eval_ldm(
    model,
    vqvae,
    scheduler,
    inferer,
    loader,
    device,
    step: int,
    writer: SummaryWriter,
    quick_test=False,
):
    print("Validating")
    model.eval()
    total_losses = 0

    pbar = tqdm(enumerate(loader), total=len(loader))
    for val_step, x in pbar:
        img = x["image"].to(device)
        with autocast(enabled=True):
            with torch.no_grad():
                # e = raw_vqvae.get_ldm_inputs(img.to(device))
                # print(img.shape)
                inputs = vqvae(img.to(device), get_ldm_inputs=True)
                noise = torch.randn_like(inputs).to(device)
                timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (inputs.shape[0],),
                    device=device,
                ).long()
                noise_prediction = model(x=inputs, timesteps=timesteps)

        loss = F.mse_loss(noise_prediction.float(), noise.float())

        total_losses = total_losses + loss * img.shape[0]

    total_losses /= len(loader.dataset)

    writer.add_scalar("loss", step)

    # save some samples
    noise = torch.randn((8, img.shape[1], img.shape[2], img.shape[3])).to(device)
    sample = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)
    fig, ax = plt.subplots(2, 4)
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(np.transpose(sample[i, ...].cpu().numpy(), (1, 2, 0)))
        plt.axis("off")
    plt.show()

    writer.add_figure(tag="samples", figure=fig, global_step=step)
    return total_losses["loss"]
