import os
from collections import OrderedDict
from pathlib import PosixPath

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..training_and_testing.util import log_reconstructions, log_ldm_sample, log_3d_ldm_sample, get_kl
import numpy as np

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_ldm(
        model,
        vqvae,
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
        best_nll: float
):
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model
    quick_test=False
    if quick_test:
        print('WARNING: just running on one batch of train and val sets.')
    # val_loss = eval_3d_ldm(
    #     model=model,
    #     vqvae=vqvae,
    #     loader=val_loader,
    #     device=device,
    #     step=len(train_loader) * start_epoch,
    #     writer=writer_val
    # )
    # print(f"epoch {start_epoch} val loss: {val_loss:.4f}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
            model=model,
            vqvae=vqvae,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            quick_test=quick_test
        )
        if (epoch + 1) % eval_freq == 0:
            val_loss, nll_per_dim = eval_ldm(
                model=model,
                vqvae=vqvae,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                quick_test=quick_test
            )

            print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'diffusion': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_nll': best_nll,
                't_sampler_history': raw_model.t_sampler._loss_history,
                't_sampler_loss_counts': raw_model.t_sampler._loss_counts
            }
            torch.save(checkpoint, str(run_dir / "checkpoint.pth"))
            if (epoch + 1) % checkpoint_every == 0:
                torch.save(checkpoint, str(run_dir / f"checkpoint_{epoch+1}.pth"))

            if val_loss <= best_loss:
                print(f"New best val loss {val_loss}")
                best_loss = val_loss
                torch.save(raw_model.state_dict(), str(run_dir / 'best_model.pth'))
            if nll_per_dim <= best_nll:
                print(f"New best nll per dim {nll_per_dim}")
                old_checkpoint = run_dir / f'best_model_nll_{best_nll:.3f}.pth'
                if old_checkpoint.exists():
                    os.remove(old_checkpoint)
                best_nll = nll_per_dim
                torch.save(raw_model.state_dict(), str(run_dir / f'best_model_nll_{best_nll:.3f}.pth'))

    print(f"Training finished!")
    print(f"Saving final model...")
    torch.save(raw_model.state_dict(), str(run_dir / 'final_model.pth'))

    return val_loss


def train_epoch_ldm(
        model,
        vqvae,
        loader,
        optimizer,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        scaler,
        quick_test: bool
):
    model.train()
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    raw_model = model.module if hasattr(model, "module") else model

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        img = x["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                #e = raw_vqvae.get_ldm_inputs(img.to(device))
                e = vqvae(img.to(device),get_ldm_inputs=True)
                e_padded = raw_vqvae.pad_ldm_inputs(e)

            loss, loss_dict = model(e_padded, crop_function=raw_vqvae.crop_ldm_inputs)
            loss = loss.mean()
            # update sampler
            raw_model.t_sampler.update_with_all_losses(loss_dict['t'], loss_dict['vlb_per_t'])
            if quick_test:
                break
        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)
        for k, v in loss_dict.items():
            if 'loss' in k:
                writer.add_scalar(k, v.mean().item(), epoch * len(loader) + step)

        pbar.set_postfix(
            {
                "epoch": epoch,
                "loss": f"{losses['loss'].item():.5f}",
                "lr": f"{get_lr(optimizer):.6f}"
            }
        )



@torch.no_grad()
def eval_ldm(
        model,
        vqvae,
        loader,
        device,
        step: int,
        writer: SummaryWriter,
        quick_test=False
):
    print('Validating')
    model.eval()
    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    raw_model = model.module if hasattr(model, "module") else model
    total_losses = OrderedDict()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for val_step, x in pbar:
        img = x["image"].to(device)
        with autocast(enabled=True):
            with torch.no_grad():
                # e = raw_vqvae.get_ldm_inputs(img.to(device))
                #print(img.shape)
                e = vqvae(img.to(device), get_ldm_inputs=True)
                e_padded = raw_vqvae.pad_ldm_inputs(e)
                loss, loss_dict = model(e_padded, crop_function = raw_vqvae.crop_ldm_inputs)

        losses = OrderedDict(loss=loss.mean())

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * img.shape[0]

        # calculate the NLL for just the first batch
        print('Calculating val NLL on 8 samples')
        mini_batch= e_padded[:8,...]
        if val_step == 0:
            kl_temp = []
            for t in tqdm(list(range(0, raw_model.num_timesteps))[::-1]):
                if t == 0:
                    continue
                t_batch = torch.full((mini_batch.shape[0],), t, device=device, dtype=torch.long)
                noise = torch.randn_like(mini_batch)
                # x_t = raw_diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
                x_t = model(mini_batch, t=t_batch, noise=noise, do_qsample='true')
                # Calculate VLB term at the current timestep
                with torch.no_grad():
                    kl = get_kl(raw_model, x_start=mini_batch, x_t=x_t, t=t_batch)
                    kl = raw_vqvae.crop_ldm_inputs(kl)
                non_batch_dims = tuple(range(kl.ndim))[1:]
                kl_temp.append(kl.sum(axis=non_batch_dims).cpu())
            nll = np.stack(kl_temp).sum(axis=0)
            latent_dim = np.prod(kl.shape[1:])
            nll_per_dim = nll / latent_dim
            for i, (n, d) in enumerate(zip(nll, nll_per_dim)):
                writer.add_scalar('NLL', n, step+i)
                writer.add_scalar('NLL_per_dim', d, step+i)
        if quick_test:
            break
    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)
    for k, v in loss_dict.items():
        if 'loss' in k:
            writer.add_scalar(k, v.mean().item(), step)

    log_3d_ldm_sample(
        diffusion_model=raw_model,
        stage1_model=raw_vqvae,
        spatial_shape=list(e_padded.shape[1:]),
        writer=writer,
        step=step,
    )

    return total_losses['loss'], nll_per_dim.mean()
