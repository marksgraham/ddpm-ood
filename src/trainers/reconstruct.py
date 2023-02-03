# import matplotlib.pyplot as plt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from generative.networks.schedulers import PNDMScheduler
from torch.nn.functional import pad

# from skimage.metrics import structural_similarity as ssim
from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.trainers.PerceptualLoss import PerceptualLoss

from .base import BaseTrainer


def shuffle(x):
    return np.transpose(x.cpu().numpy(), (1, 2, 0))


class Reconstruct(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        # set up dirs
        out_dir = self.run_dir / "ood"
        out_dir.mkdir(exist_ok=True)

        # set up loaders
        self.val_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.validation_ids,
            validation_ids=args.validation_ids,
            augmentation=bool(args.augmentation),
            only_val=True,
            num_workers=args.num_workers,
            num_val_workers=args.num_workers,
            cache_data=False,
            drop_last=bool(args.drop_last),
            first_n=int(args.first_n_val) if args.first_n_val else args.first_n_val,
            is_grayscale=bool(args.is_grayscale),
        )

        self.in_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.in_ids,
            validation_ids=args.in_ids,
            augmentation=bool(args.augmentation),
            only_val=True,
            num_workers=args.num_workers,
            num_val_workers=args.num_workers,
            cache_data=False,
            drop_last=bool(args.drop_last),
            first_n=int(args.first_n) if args.first_n else args.first_n,
            is_grayscale=bool(args.is_grayscale),
        )

    def get_scores(self, loader, dataset_name):
        results = []
        pl = PerceptualLoss(
            dimensions=2,
            include_pixel_loss=False,
            is_fake_3d=False,
            lpips_normalize=True,
            spatial=False,
        ).to(self.device)

        self.model.eval()
        pndm_scheduler = PNDMScheduler(num_train_timesteps=1000, skip_prk_steps=True)
        pndm_scheduler.set_timesteps(100)
        pndm_timesteps = pndm_scheduler.timesteps
        with torch.no_grad():
            for batch in loader:
                t1 = time.time()
                images = batch["image"].to(self.device)
                # loop over different values to reconstruct from
                for t_start in reversed(pndm_timesteps)[1:]:
                    noise = torch.randn_like(images).to(self.device)
                    start_timesteps = torch.Tensor([t_start] * images.shape[0]).long()

                    reconstructions = pndm_scheduler.add_noise(
                        original_samples=images, noise=noise, timesteps=start_timesteps
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
                    reconstructions.clamp_(0, 1)
                    # compute similarity
                    perceptual_difference = pl(
                        pad(images, (2, 2, 2, 2)),
                        pad(
                            reconstructions,
                            (2, 2, 2, 2),
                        ),
                    )
                    # ssim_metric = 1 - ssim(
                    #                 images.squeeze().cpu().numpy(),
                    #                 reconstructions[b, ...].clamp(0, 1).squeeze().cpu().numpy(),
                    #             )

                    mse_metric = torch.square(images - reconstructions).mean(axis=(1, 2, 3))
                    for b in range(images.shape[0]):
                        filename = batch["image_meta_dict"]["filename_or_obj"][b]
                        stem = Path(filename).stem.replace(".nii", "").replace(".gz", "")
                        results.append(
                            {
                                "filename": stem,
                                "type": dataset_name,
                                "t": t_start,
                                "perceptual_difference": perceptual_difference[b].item(),
                                # "ssim": ssim_metric,
                                "mse": mse_metric[b].item(),
                            }
                        )
                    # # plot
                    # fig, ax = plt.subplots(8, 2, figsize=(2, 8))
                    # for i in range(8):
                    #     plt.subplot(8, 2, i * 2 + 1)
                    #     plt.imshow(shuffle(images[i, ...]), vmin=0, vmax=1, cmap="gray")
                    #     plt.axis("off")
                    #     plt.subplot(8, 2, i * 2 + 2)
                    #     plt.imshow(shuffle(reconstructions[i, ...]), vmin=0, vmax=1, cmap="gray")
                    #     plt.axis("off")
                    # plt.suptitle(f"Recon from: {t_start}")
                    # plt.tight_layout()
                    # plt.show()
                t2 = time.time()
                print(f"Took {t2-t1}s for a batch size of {images.shape[0]}")
        return results

    def reconstruct(self, args):
        if bool(args.run_val):
            results_list = self.get_scores(self.val_loader, "val")
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(self.out_dir / "results_val.csv")

        if bool(args.run_in):
            results_list = self.get_scores(self.in_loader, "in")
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
                        cache_data=False,
                        drop_last=bool(args.drop_last),
                        first_n=int(args.first_n) if args.first_n else args.first_n,
                        is_grayscale=bool(args.is_grayscale),
                        add_vflip=True,
                    )
                    dataset_name = Path(out).stem.split("_")[0] + "_vflip"
                    results_list = self.get_scores(out_loader, "out")
                    results_df = pd.DataFrame(results_list)
                    results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")
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
                        cache_data=False,
                        drop_last=bool(args.drop_last),
                        first_n=int(args.first_n) if args.first_n else args.first_n,
                        is_grayscale=bool(args.is_grayscale),
                        add_hflip=True,
                    )
                    dataset_name = Path(out).stem.split("_")[0] + "_vflip"
                    results_list = self.get_scores(out_loader, "out")
                    results_df = pd.DataFrame(results_list)
                    results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")
                else:
                    out_loader = get_training_data_loader(
                        batch_size=args.batch_size,
                        training_ids=out,
                        validation_ids=out,
                        augmentation=bool(args.augmentation),
                        only_val=True,
                        num_workers=args.num_workers,
                        num_val_workers=args.num_workers,
                        cache_data=False,
                        drop_last=bool(args.drop_last),
                        first_n=int(args.first_n) if args.first_n else args.first_n,
                        is_grayscale=bool(args.is_grayscale),
                    )
                    dataset_name = Path(out).stem.split("_")[0] + "_vflip"
                    results_list = self.get_scores(out_loader, "out")
                    results_df = pd.DataFrame(results_list)
                    results_df.to_csv(self.out_dir / f"results_{dataset_name}.csv")
