from pathlib import PosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai import transforms
from monai.data import CacheDataset, Dataset
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)

def get_data_dicts(
        ids_path: str,
        shuffle: bool = False,
):
    """ Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep=",")
    if shuffle:
        df = df.sample(frac=1, random_state=1)
    df = list(df)
    data_dicts = []
    for row in df:
        data_dicts.append(
            {
                "image": (row)
            }
        )


    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def get_training_data_loader(
        batch_size: int,
        training_ids: str,
        validation_ids: str,
        only_val: bool = False,
        augmentation: bool = True,
        drop_last: bool = False,
        num_workers: int = 8,
        num_val_workers: int=3,
        cache_data=True,
        first_n=None,
        is_grayscale=False,
        add_vflip=False,
        add_hflip=False,
):
    # Define transformations
    val_transforms = transforms.Compose([
        transforms.LoadImaged(keys=['image']),
        transforms.EnsureChannelFirstd(keys=['image']) if is_grayscale else lambda x: x,
        transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        transforms.RandFlipD(keys=['image'],spatial_axis=0,prob=1.0) if add_vflip else lambda x: x,
        transforms.RandFlipD(keys=['image'],spatial_axis=1,prob=1.0) if add_hflip else lambda x: x,
        transforms.ToTensord(keys=['image'])
    ])

    if augmentation:
        train_transforms = transforms.Compose([
            transforms.LoadImaged(keys=['image']),
            transforms.ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
            transforms.EnsureChannelFirstd(keys=['image']) if is_grayscale else lambda x: x,
            transforms.ToTensord(keys=['image']),
        ])
    else:
        train_transforms = val_transforms


    val_dicts = get_data_dicts(
        validation_ids,
        shuffle=False,
    )
    if first_n:
        val_dicts = val_dicts[:first_n]
    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=val_transforms,
        )
    print(val_ds[0]['image'].shape)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_val_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(
        training_ids,
        shuffle=False,
    )
    if first_n:
        train_dicts = train_dicts[:first_n]
    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False
    )

    return train_loader, val_loader


# ----------------------------------------------------------------------------------------------------------------------
# TEST TIME FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def get_kl(model, x_start, x_t, t):
    true_mean, _, true_log_variance_clipped = model.q_posterior(x_start=x_start, x_t=x_t, t=t)
    model_mean, _, posterior_log_variance = model.p_mean_variance(x_t, t, clip_denoised=False)
    kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, posterior_log_variance)
    return kl


# ----------------------------------------------------------------------------------------------------------------------
# LOGS
# ----------------------------------------------------------------------------------------------------------------------
def get_figure(
        img,
        recons,
):
    img_npy_0 = np.clip(
        a=img[0, 0, :, :].cpu().numpy(),
        a_min=0,
        a_max=1
    )
    recons_npy_0 = np.clip(
        a=recons[0, 0, :, :].cpu().numpy(),
        a_min=0,
        a_max=1
    )
    img_npy_1 = np.clip(
        a=img[1, 0, :, :].cpu().numpy(),
        a_min=0,
        a_max=1
    )
    recons_npy_1 = np.clip(
        a=recons[1, 0, :, :].cpu().numpy(),
        a_min=0,
        a_max=1
    )


    img_row_0 = np.concatenate(
        (
            img_npy_0,
            recons_npy_0,
            img_npy_1,
            recons_npy_1,
        ),
        axis=1
    )


    img = np.concatenate(
        (
            img_row_0,
            img_row_0,
        ),
        axis=0
    )

    fig = plt.figure(dpi=300)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    return fig

def get_figure_3d(
        img,
        recons,
):
    if img.ndim == 5:
        sl1 = np.s_[0,0,:,:,90]
        sl2 = np.s_[1,0,:,:,90]
    elif img.ndim==4:
        sl1=np.s_[0,:,:,:]
        sl2=np.s_[1,:,:,:]
    img_npy_0 = np.clip(
        a=img[sl1].cpu().numpy(),
        a_min=0,
        a_max=1
    )
    recons_npy_0 = np.clip(
        a=recons[sl1].cpu().numpy(),
        a_min=0,
        a_max=1
    )
    img_npy_1 = np.clip(
        a=img[sl2].cpu().numpy(),
        a_min=0,
        a_max=1
    )
    recons_npy_1 = np.clip(
        a=recons[sl2].cpu().numpy(),
        a_min=0,
        a_max=1
    )
    if img.ndim == 5:
        img_row_0 = np.concatenate(
            (
                img_npy_0,
                recons_npy_0,
                img_npy_1,
                recons_npy_1,
            ),
            axis=1
        )
        img = np.concatenate(
        (
            img_row_0,
            img_row_0,
        ),
        axis=0
    )
        fig = plt.figure(dpi=300)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    elif img.ndim==4:
        img_row_0 = np.concatenate(
            (
                img_npy_0,
                recons_npy_0,
                img_npy_1,
                recons_npy_1,
            ),
            axis=2
        )
        img = np.concatenate(
            (
                img_row_0,
                img_row_0,
            ),
            axis=1
        )
        fig = plt.figure(dpi=300)
        plt.imshow(np.transpose(img,(1,2,0)), cmap="gray")
        plt.axis("off")

    return fig

def log_reconstructions(
        img: torch.Tensor,
        recons: torch.Tensor,
        writer: SummaryWriter,
        step: int,
):
    fig = get_figure(
        img,
        recons,
    )
    writer.add_figure(f"RECONSTRUCTION", fig, step)

def log_reconstructions_3d(
        img: torch.Tensor,
        recons: torch.Tensor,
        writer: SummaryWriter,
        step: int,
        name: str = 'Reconstructions'
):
    fig = get_figure_3d(
        img,
        recons,
    )
    writer.add_figure(name, fig, step)

def log_ldm_sample(
        diffusion_model,
        stage1_model,
        spatial_shape,
        writer: SummaryWriter,
        step: int,
):
    sample_shape = [8, ] + spatial_shape
    latent_vectors = diffusion_model.p_sample_loop(sample_shape, return_intermediates=False)

    with torch.no_grad():
        x_hat = stage1_model.reconstruct_ldm_outputs(latent_vectors)

    log_reconstructions(
        img=x_hat[:4],
        recons=x_hat[4:],
        writer=writer,
        step=step,
    )

def log_3d_ldm_sample(
        diffusion_model,
        stage1_model,
        spatial_shape,
        writer: SummaryWriter,
        step: int,
):
    sample_shape = [8, ] + spatial_shape
    latent_vectors_padded = diffusion_model.p_sample_loop(sample_shape, return_intermediates=False)
    latent_vectors = stage1_model.crop_ldm_inputs(latent_vectors_padded)
    with torch.no_grad():
        x_hat = stage1_model.reconstruct_ldm_outputs(latent_vectors)

    log_reconstructions_3d(
        img=x_hat[:4],
        recons=x_hat[4:],
        writer=writer,
        step=step,
        name='Samples'
    )

