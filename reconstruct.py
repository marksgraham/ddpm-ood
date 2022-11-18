import argparse
import warnings
from pathlib import Path

import pandas as pd
import torch
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf

from src.models.vqvae_2d import BaselineVQVAE2D
from src.models.vqvae_dummy import DummyVQVAE
from src.models.ddpm_2d import DDPM
from src.models.plms import PLMSSampler
from src.training_and_testing.util import get_training_data_loader
from tqdm import tqdm
import numpy as np
import os
from src.training_and_testing.PerceptualLoss import PerceptualLoss
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import pad as torchpad
warnings.filterwarnings("ignore")
from time import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location for models.")
    parser.add_argument("--model_name", help="Name of model.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--in_ids", help="Location of file with inlier ids.")
    parser.add_argument("--out_ids", help="List of location of file with outlier ids.")
    parser.add_argument("--config_vqvae_file", default='None', help="Location of VQ-VAE config. None if not training a latent diffusion model.")
    parser.add_argument("--vqvae_checkpoint", help="Path to checkpoint file.")
    parser.add_argument("--config_diffusion_file", help="Location of config.")
    parser.add_argument("--vqvae_uri", help="Path readable by load_model.")

    # inference param
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--augmentation", type=int, default=0, help="Use of augmentation, 1 (True) or 0 (False).")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--first_n_val", default=None, help="Only run on the first n samples from the val dataset.")
    parser.add_argument("--first_n", default=None, help="Only run on the first n samples from each dataset.")
    parser.add_argument("--eval_checkpoint", default=None, help="Select a specific checkpoint to evaluate on.")
    parser.add_argument("--drop_last", default=False, help="Drop last non-complete batch..")
    parser.add_argument("--is_grayscale", type=int, default=0, help="Is data grayscale.")
    parser.add_argument("--run_val", type=int, default=1, help="Run reconstructions on val set.")
    parser.add_argument("--run_in", type=int, default=1, help="Run reconstructions on in set.")
    parser.add_argument("--run_out",type=int, default=1, help="Run reconstructions on out set.")

    # sampling options
    parser.add_argument("--num_inference_steps", type=int, default=100,
                        help="Number of inference steps to use with the PLMS sampler.")
    parser.add_argument("--inference_skip_factor", type=int, default=1,
                        help="Perform fewer reconstructions by skipping some of the t-values as starting points.")


    args = parser.parse_args()
    return args

def main(args):
    set_determinism(seed=args.seed)
    print_config()
    run_dir = Path(args.output_dir) / args.model_name
    if args.eval_checkpoint:
            checkpoint_path = Path(args.eval_checkpoint)
    else:
        checkpoint_paths = list(run_dir.glob('best_model_nll*.pth'))
        print(run_dir, checkpoint_paths)
        assert len(checkpoint_paths) == 1
        checkpoint_path = checkpoint_paths[0]
    print(f'Running on checkpoint {checkpoint_path}')
    if run_dir.exists() and (checkpoint_path).exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")



    print("Getting data...")
    print(f"Using first_n_val argument for for val: {args.first_n_val}")

    val_loader = get_training_data_loader(
        batch_size=args.batch_size,
        training_ids=args.validation_ids,
        validation_ids=args.validation_ids,
        augmentation=bool(args.augmentation),
        only_val=True,
        num_workers=args.num_workers,
        num_val_workers=args.num_workers,
        cache_data=False,
        drop_last=bool(args.drop_last),
        first_n= int(args.first_n_val) if args.first_n_val else args.first_n_val,
        is_grayscale=bool(args.is_grayscale)
    )

    in_loader = get_training_data_loader(
        batch_size=args.batch_size,
        training_ids=args.in_ids,
        validation_ids=args.in_ids,
        augmentation=bool(args.augmentation),
        only_val=True,
        num_workers=args.num_workers,
        num_val_workers=args.num_workers,
        cache_data=False,
        drop_last=bool(args.drop_last),
        first_n= int(args.first_n) if args.first_n else args.first_n,
        is_grayscale=bool(args.is_grayscale)
    )

    # Load VQVAE to produce the encoded samples
    if args.config_vqvae_file !='None':
        config_vqvae = OmegaConf.load(args.config_vqvae_file)
        vqvae = BaselineVQVAE2D(**config_vqvae["stage1"])
        if os.environ['HOME'] == '/root':
            checkpoint = torch.load(args.vqvae_checkpoint)
        else:
            checkpoint = torch.load(args.vqvae_checkpoint)
        print(f'Loaded VQVAE checkpoint  {args.vqvae_checkpoint}')
        vqvae.load_state_dict(checkpoint['network'])
        vqvae.eval()
    else:
        vqvae = DummyVQVAE()

    # Load diffusion model
    print("Creating model...")
    config_ldm = OmegaConf.load(args.config_diffusion_file)
    diffusion = DDPM(**config_ldm["ldm"].get("params", dict()))

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        vqvae = torch.nn.DataParallel(vqvae)
        diffusion = torch.nn.DataParallel(diffusion)


    vqvae = vqvae.to(device)
    diffusion = diffusion.to(device)

    if resume:
        print(f"Using checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        if 'diffusion' in checkpoint.keys():
            checkpoint = checkpoint['diffusion']
        if torch.cuda.device_count() > 1:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if 'module' not in k:
                    k = 'module.' + k
                else:
                    k = k.replace('features.module.', 'module.features.')
                new_state_dict[k] = v
            diffusion.load_state_dict(new_state_dict)
        else:
            diffusion.load_state_dict(checkpoint)
    else:
        raise FileExistsError(f'No checkpoint {checkpoint_path}')


    raw_vqvae = vqvae.module if hasattr(vqvae, "module") else vqvae
    raw_diffusion = diffusion.module if hasattr(diffusion, "module") else diffusion
    out_dir = run_dir / 'ood'
    out_dir.mkdir(exist_ok=True)

    diffusion_plms = PLMSSampler(diffusion)
    raw_diffusion.device = device
    diffusion_plms.make_schedule(ddim_num_steps=args.num_inference_steps)

    t_vals = diffusion_plms.ddim_timesteps[::args.inference_skip_factor]
    total_steps = 0
    for t in t_vals:
        steps_for_this_t = diffusion_plms.ddim_timesteps[diffusion_plms.ddim_timesteps<=t]
        total_steps+=len(steps_for_this_t)
    print(f'Using num_inference steps {args.num_inference_steps} and a total of {len(t_vals)} different reconstruction'
          f' t-values, giving a total of {total_steps} model evaluations')

    def get_scores(dataloader, dataset_name, out_dir):
        results_list = []
        pl = PerceptualLoss(dimensions=2, include_pixel_loss=False, is_fake_3d=False, lpips_normalize=True, spatial=False)
        pl_spatial = PerceptualLoss(dimensions=2, include_pixel_loss=False, is_fake_3d=False, lpips_normalize=True, spatial=True)
        n=0
        with torch.no_grad():
            print(dataset_name)
            for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
                t1 = time()
                image = batch['image'].cuda()
                latent = vqvae(image, get_ldm_inputs=True)
                latent_padded = raw_vqvae.pad_ldm_inputs(latent)
                batch_size = latent_padded.shape[0]
                n += batch['image'].shape[0]
                for idx, t in enumerate(t_vals):
                    t_batch_upper = torch.full((batch_size,), t_vals[idx], device=device, dtype=torch.long)
                    noise = torch.randn_like(latent_padded)
                    x_t_upper = raw_diffusion(latent_padded, t=t_batch_upper, noise=noise, do_qsample='true')
                    latent_denoised = x_t_upper

                    latent_denoised,_ = diffusion_plms.sample(S=args.num_inference_steps,
                                                batch_size=1,
                                                shape=list(in_loader.dataset[0]['image'].shape),
                                                timesteps=diffusion_plms.ddim_timesteps[diffusion_plms.ddim_timesteps<=t],
                                                x_T=latent_denoised,
                                                verbose=False)

                    for b in range(batch_size):
                        if args.config_vqvae_file =='None':
                            if x_t_upper.shape[-2:][0] < 32:
                                # images < 32*32 are grayscale FashionMNIST/MNist
                                perceptual_difference_spatial = pl_spatial(torchpad(image[b,None,...].cpu(),(2,2,2,2)),
                                                           torchpad(latent_denoised[b,None,...].clamp(0,1).cpu(),(2,2,2,2))
                                                           ).numpy()
                                perceptual_difference = perceptual_difference_spatial.mean()
                                perceptual_difference_spatial=perceptual_difference_spatial[:,:,2:-2,2:-2]

                                ssim_metric = 1 - ssim(image[b, ...].squeeze().cpu().numpy(),
                                                       latent_denoised[b, ...].clamp(0, 1).squeeze().cpu().numpy())
                            else:
                                # all other images are 32x32 and colour
                                #perceptual_difference = pl(image[b,None,...].cpu(),latent_denoised[b,None,...].cpu()).item()
                                perceptual_difference_spatial = pl_spatial(image[b,None,...].cpu(),latent_denoised[b,None,...].cpu()).numpy()
                                perceptual_difference = perceptual_difference_spatial.mean()
                                ssim_metric = 1 - ssim(np.transpose(image[b, ...].squeeze().cpu().numpy(), (1, 2, 0)),
                                                       np.transpose(
                                                           latent_denoised[b, ...].clamp(0, 1).squeeze().cpu().numpy(),
                                                           (1, 2, 0)), multichannel=True)

                        se = np.mean(np.square(image[b,...].squeeze().cpu().numpy()-latent_denoised[b,...].clamp(0,1).squeeze().cpu().numpy()),axis=0)
                        mse_metric = np.mean(se)
                        # save val-set metrics for z-scoring
                        if 'val' in dataset_name:
                            if batch_idx == 0 and idx == 0 and b == 0:
                                shape = (len(t_vals),)+perceptual_difference_spatial.shape[2:]
                                shape_mse = (len(t_vals),)+perceptual_difference_spatial.shape[2:]
                                all_perceptual_difference_x = np.zeros(shape)
                                all_perceptual_difference_x2 = np.zeros(shape)
                                all_mse_x = np.zeros(shape)
                                all_mse_x2 = np.zeros(shape)
                            all_perceptual_difference_x[idx,:] +=perceptual_difference_spatial[0,0,...]
                            all_perceptual_difference_x2[idx,:] += np.square(perceptual_difference_spatial[0,0,...])
                            all_mse_x[idx,:] += se
                            all_mse_x2[idx,:] += np.square(se)
                        filename = batch['image_meta_dict']['filename_or_obj'][b]
                        stem = Path(filename).stem.replace('.nii', '').replace('.gz', '')
                        result_dict={'filename':stem,
                                     'type': dataset_name,
                                     't':t,
                                     'perceptual_difference': perceptual_difference,
                                     'ssim': ssim_metric,
                                     'mse': mse_metric,
                                     }
                        results_list.append(result_dict)
                       # print(result_dict)
                t2 = time()
                print(f'Took {t2 - t1}s for a batch size of {batch_size}')
            if 'val' in dataset_name:
                # compute mean/std for metrics
                all_perceptual_difference_mean = all_perceptual_difference_x/n
                all_perceptual_difference_std = np.sqrt(all_perceptual_difference_x2/n - np.square(all_perceptual_difference_mean))
                all_mse_mean = all_mse_x/n
                all_mse_std = np.sqrt(all_mse_x2/n - np.square(all_mse_mean))
                np.save(out_dir / 'lpips_mean', all_perceptual_difference_mean)
                np.save(out_dir / 'lpips_std', all_perceptual_difference_std)
                np.save(out_dir / 'mse_mean', all_mse_mean)
                np.save(out_dir / 'mse_std', all_mse_std)

            return results_list

    if bool(args.run_val):
        results_list = get_scores(val_loader, 'val', out_dir)
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(out_dir / 'results_val.csv')

    if bool(args.run_in):
        results_list = get_scores(in_loader, 'in', out_dir)
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(out_dir / 'results_in.csv')

    if bool(args.run_out):
        for out in args.out_ids.split(','):
            print(out)
            if 'vflip' in out:
                out = out.replace('_vflip','')
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
                    add_vflip=True
                )
                dataset_name = Path(out).stem.split('_')[0] + '_vflip'
                results_list = get_scores(out_loader, 'out', out_dir)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(out_dir / f'results_{dataset_name}.csv')
            elif 'hflip' in out:
                out = out.replace('_hflip','')
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
                    add_hflip=True
                )
                dataset_name = Path(out).stem.split('_')[0] + '_hflip'
                results_list = get_scores(out_loader, 'out', out_dir)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(out_dir / f'results_{dataset_name}.csv')
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
                    is_grayscale=bool(args.is_grayscale)
                )
                dataset_name = Path(out).stem.split('_')[0]
                results_list = get_scores(out_loader, 'out', out_dir)
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(out_dir / f'results_{dataset_name}.csv')
if __name__ == "__main__":
    args = parse_args()
    main(args)
