# import matplotlib.pyplot as plt

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from generative.networks.schedulers import PNDMScheduler
from torch.cuda.amp import autocast
from torch.nn.functional import pad

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.trainers.PerceptualLoss import PerceptualLoss

from .base import BaseTrainer


def shuffle(x):
    return np.transpose(x.cpu().numpy(), (1, 2, 0))


class Reconstruct(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        if not self.found_checkpoint:
            raise FileNotFoundError("Failed to find a saved model checkpoint.")
        # set up dirs
        self.out_dir = self.run_dir / "ood"
        self.out_dir.mkdir(exist_ok=True)

        # set up loaders
        self.val_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.validation_ids,
            validation_ids=args.validation_ids,
            augmentation=bool(args.augmentation),
            only_val=True,
            num_workers=args.num_workers,
            num_val_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            drop_last=bool(args.drop_last),
            first_n=int(args.first_n_val) if args.first_n_val else args.first_n_val,
            is_grayscale=bool(args.is_grayscale),
            image_size=self.image_size,
            image_roi=args.image_roi,
        )

        self.in_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.in_ids,
            validation_ids=args.in_ids,
            augmentation=bool(args.augmentation),
            only_val=True,
            num_workers=args.num_workers,
            num_val_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            drop_last=bool(args.drop_last),
            first_n=int(args.first_n) if args.first_n else args.first_n,
            is_grayscale=bool(args.is_grayscale),
            image_size=self.image_size,
            image_roi=args.image_roi,
        )

    def get_scores(self, loader, dataset_name, inference_skip_factor):
        if dist.is_initialized():
            # temporarily enable logging on every node
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{dist.get_rank()}: {dataset_name}")
        else:
            print(f"{dataset_name}")

        results = []
        pl = PerceptualLoss(
            dimensions=self.spatial_dimension,
            include_pixel_loss=False,
            is_fake_3d=True if self.spatial_dimension == 3 else False,
            lpips_normalize=True,
            spatial=False,
        ).to(self.device)
        # ms_ssim = MSSSIM(
        #     data_range=torch.tensor(1.0).to(self.device),
        #     spatial_dims=2,
        #     weights=torch.Tensor([0.0448, 0.2856]).to(self.device),
        # )

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                pndm_scheduler = PNDMScheduler(
                    num_train_timesteps=1000,
                    skip_prk_steps=True,
                    prediction_type=self.prediction_type,
                    beta_schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                )
                pndm_scheduler.set_timesteps(100)
                pndm_timesteps = pndm_scheduler.timesteps
                pndm_start_points = reversed(pndm_timesteps)[1::inference_skip_factor]

                t1 = time.time()
                images_original = batch["image"].to(self.device)
                images = self.vqvae_model.encode_stage_2_inputs(images_original)
                if self.do_latent_pad:
                    images = F.pad(input=images, pad=self.latent_pad, mode="constant", value=0)
                # loop over different values to reconstruct from
                for t_start in pndm_start_points:
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(self.device)
                        start_timesteps = torch.Tensor([t_start] * images.shape[0]).long()

                        reconstructions = pndm_scheduler.add_noise(
                            original_samples=images * self.b_scale,
                            noise=noise,
                            timesteps=start_timesteps,
                        )
                        # perform reconstruction
                        for step in pndm_timesteps[pndm_timesteps <= t_start]:
                            timesteps = torch.Tensor([step] * images.shape[0]).long()
                            model_output = self.model(
                                reconstructions, timesteps=timesteps.to(self.device)
                            )
                            # 2. compute previous image: x_t -> x_t-1
                            reconstructions, _ = pndm_scheduler.step(
                                model_output, step, reconstructions
                            )
                    # try clamping the reconstructions
                    if self.do_latent_pad:
                        reconstructions = F.pad(
                            input=reconstructions,
                            pad=self.inverse_latent_pad,
                            mode="constant",
                            value=0,
                        )
                    reconstructions = self.vqvae_model.decode_stage_2_outputs(reconstructions)
                    reconstructions = reconstructions / self.b_scale
                    reconstructions.clamp_(0, 1)
                    # compute similarity
                    if self.spatial_dimension == 2:
                        if images_original.shape[3] == 28:
                            perceptual_difference = pl(
                                pad(images_original, (2, 2, 2, 2)),
                                pad(
                                    reconstructions,
                                    (2, 2, 2, 2),
                                ),
                            )
                        else:
                            perceptual_difference = pl(images_original, reconstructions)
                    else:
                        # in 3D need to calculate perceptual difference for each batch item seperately for now
                        perceptual_difference = torch.empty(images.shape[0])
                        for b in range(images.shape[0]):
                            perceptual_difference[b] = pl(
                                images_original[b, None, ...], reconstructions[b, None, ...]
                            )
                    non_batch_dims = tuple(range(images_original.dim()))[1:]
                    mse_metric = torch.square(images_original - reconstructions).mean(
                        axis=non_batch_dims
                    )
                    for b in range(images.shape[0]):
                        filename = batch["image_meta_dict"]["filename_or_obj"][b]
                        stem = Path(filename).stem.replace(".nii", "").replace(".gz", "")

                        results.append(
                            {
                                "filename": stem,
                                "type": dataset_name,
                                "t": t_start.item(),
                                "perceptual_difference": perceptual_difference[b].item(),
                                "mse": mse_metric[b].item(),
                            }
                        )
                    # plot
                    if not dist.is_initialized():
                        import matplotlib.pyplot as plt

                        n_rows = min(images.shape[0], 8)
                        fig, ax = plt.subplots(n_rows, 2, figsize=(2, n_rows))
                        for i in range(n_rows):
                            image_slice = (
                                np.s_[i, :, :, images_original.shape[4] // 2]
                                if self.spatial_dimension == 3
                                else np.s_[i, :, :]
                            )
                            plt.subplot(n_rows, 2, i * 2 + 1)
                            plt.imshow(
                                shuffle(images_original[image_slice]), vmin=0, vmax=1, cmap="gray"
                            )
                            plt.axis("off")
                            plt.subplot(n_rows, 2, i * 2 + 2)
                            plt.imshow(
                                shuffle(reconstructions[image_slice]), vmin=0, vmax=1, cmap="gray"
                            )
                            # plt.title(f"{mse_metric[i].item():.3f}")
                            plt.title(f"{perceptual_difference[i].item():.3f}")
                            plt.axis("off")
                        plt.suptitle(f"Recon from: {t_start}")
                        plt.tight_layout()
                        plt.show()
                t2 = time.time()
                if dist.is_initialized():
                    print(f"{dist.get_rank()}: Took {t2-t1}s for a batch size of {images.shape[0]}")
                else:
                    print(f"Took {t2-t1}s for a batch size of {images.shape[0]}")
        # gather results from all processes
        if dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, results)
            # un-nest
            all_results = [item for sublist in all_results for item in sublist]
            # return to only logging on the first device
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f
            return all_results
        else:
            return results

    def reconstruct(self, args):
        if bool(args.run_val):
            results_list = self.get_scores(self.val_loader, "val", args.inference_skip_factor)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_val.csv")

        if bool(args.run_in):
            results_list = self.get_scores(self.in_loader, "in", args.inference_skip_factor)

            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_in.csv")

        if bool(args.run_out):
            for out in args.out_ids.split(","):
                print(out)
                if "vflip" in out:
                    out = out.replace("_vflip", "")
                    out_loader = get_training_data_loader(
                        batch_size=args.batch_size,
                        training_ids=out,
                        validation_ids=out,
                        augmentation=bool(args.augmentation),
                        only_val=True,
                        num_workers=args.num_workers,
                        num_val_workers=args.num_workers,
                        cache_data=bool(args.cache_data),
                        drop_last=bool(args.drop_last),
                        first_n=int(args.first_n) if args.first_n else args.first_n,
                        is_grayscale=bool(args.is_grayscale),
                        image_size=self.image_size,
                        add_vflip=True,
                        image_roi=args.image_roi,
                    )
                    dataset_name = Path(out).stem.split("_")[0] + "_vflip"

                elif "hflip" in out:
                    out = out.replace("_hflip", "")
                    out_loader = get_training_data_loader(
                        batch_size=args.batch_size,
                        training_ids=out,
                        validation_ids=out,
                        augmentation=bool(args.augmentation),
                        only_val=True,
                        num_workers=args.num_workers,
                        num_val_workers=args.num_workers,
                        cache_data=bool(args.cache_data),
                        drop_last=bool(args.drop_last),
                        first_n=int(args.first_n) if args.first_n else args.first_n,
                        is_grayscale=bool(args.is_grayscale),
                        image_size=self.image_size,
                        add_hflip=True,
                        image_roi=args.image_roi,
                    )
                    dataset_name = Path(out).stem.split("_")[0] + "_hflip"

                else:
                    out_loader = get_training_data_loader(
                        batch_size=args.batch_size,
                        training_ids=out,
                        validation_ids=out,
                        augmentation=bool(args.augmentation),
                        only_val=True,
                        num_workers=args.num_workers,
                        num_val_workers=args.num_workers,
                        cache_data=bool(args.cache_data),
                        drop_last=bool(args.drop_last),
                        first_n=int(args.first_n) if args.first_n else args.first_n,
                        is_grayscale=bool(args.is_grayscale),
                        image_size=self.image_size,
                        image_roi=args.image_roi,
                    )
                    dataset_name = Path(out).stem.split("_")[0]
                results_list = self.get_scores(out_loader, "out", args.inference_skip_factor)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")
