<h1 align="center">Denoising diffusion models for out-of-distribution detection</h1>
<p align="center">
Perform reconstruction-based out-of-distribution detection with DDPMs.
</p>

<p align="center">
  <img width="800" height="300" src="https://user-images.githubusercontent.com/7947315/233470531-df6437d7-e277-4147-96a0-6aa354cf2ef4.svg">
</p>


## Intro

This codebase contains the code to perform unsupervised out-of-distribution detection with diffusion models.
It supports the use of DDPMs as well as Latent Diffusion Models (LDM) for dealing with higher dimensional 2D or 3D data.
It is based on work published in [1] and [2].

[1] [Denoising diffusion models for out-of-distribution detection, CVPR VAND Workshop 2023](https://arxiv.org/abs/2211.07740)

[2] [Unsupervised 3D out-of-distribution detection with latent diffusion models, MICCAI 2023]()

## Setup

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.8) and then install the required packages:

```pip install -r requirements.txt```

You can also build the docker image
```bash
cd docker/
bash create_docker_image.sh
```

### Setup paths
Select where you want your data and model outputs stored.
```
data_root=/root/for/downloaded/dataset
output_root=/root/for/saved/models
```

## Run with DDPM
We'll use the example of FashionMNIST as an in-distribution dataset and [SVHN,CIFAR10, CelebA] as out-of-distribution datasets.
### Download and process datasets
```bash
python src/data/get_computer_vision_datasets.py --data_root=${data_root}
```
N.B. If the error "The daily quota of the file img_align_celeba.zip is exceeded and it can't be downloaded" is thrown,
you need to download these files manually from the GDrive and place them in `${data_root}/CelebA/raw/`,
[see here](https://github.com/pytorch/vision/issues/1920#issuecomment-852237902). You can then run

```bash
python get_datasets.py --data_root=${data_root} --download_celeba=False
```

To use your own data, you just need to provide separate csvs containing paths for the train/val/test splits.

### Train models
Examples here use FashionMNIST as the in-distribution dataset. Commands for other datasets are given
in [README_additional.md](README_additional.md).

```bash
python train_ddpm.py \
--output_dir=${output_root} \
--model_name=fashionmnist \
--training_ids=${data_root}/data_splits/FashionMNIST_train.csv \
--validation_ids=${data_root}/data_splits/FashionMNIST_val.csv \
--is_grayscale=1 \
--n_epochs=300 \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195
```

You can track experiments in tensorboard
```bash
tensorboard --logdir=${output_root}
```

The code is DistributedDataParallel (DDP) compatible. To train on e.g. 2 GPUs:

```bash
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
train_ddpm.py \
--output_dir=${output_root} \
--model_name=fashionmnist \
--training_ids=${data_root}/data_splits/FashionMNIST_train.csv \
--validation_ids=${data_root}/data_splits/FashionMNIST_val.csv \
--is_grayscale=1 \
--n_epochs=300 \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195
```

### Reconstruct data

```bash
python reconstruct.py \
--output_dir=${output_root} \
--model_name=fashionmnist \
--validation_ids=${data_root}/data_splits/FashionMNIST_val.csv \
--in_ids=${data_root}/data_splits/FashionMNIST_test.csv \
--out_ids=${data_root}/data_splits/MNIST_test.csv,${data_root}/data_splits/FashionMNIST_vflip_test.csv,${data_root}/data_splits/FashionMNIST_hflip_test.csv \
--is_grayscale=1 \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195 \
--num_inference_steps=100 \
--inference_skip_factor=4 \
--run_val=1 \
--run_in=1 \
--run_out=1
```
The arg `inference_skip_factor` controls the amount of t starting points that are skipped during reconstruction.
This table shows the relationship between values of `inference_skip_factor` and the number of reconstructions, as needed
to reproduce results in Supplementary Table 4 (for max_t=1000).

| **inference_skip_factor:** | 1   | 2   | 3   | 4   | 5   | 8   | 16  | 32  | 64  |
|------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| **num_reconstructions:**   | 100 | 50  | 34  | 25  | 20  | 13  | 7   | 4   | 2   |

N.B. For a quicker run, you can choose to only reconstruct a subset of the validation set with e.g. `--first_n_val=1000`
or a subset of the in/out datasets with `--first_n=1000`


### Classify samples as OOD
```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=fashionmnist
```

## Run with LDM
We'll use the 3D Medical Decathlon Dataset here. In this example we'll use the BraTS dataset as the in-distribution dataset,
and the other 9 datasets as out-of-distribution datasets.

### Download and process datasets
```bash
python src/data/get_decathlon_datasets.py --data_root=${data_root}/Decathlon
```
### Train VQVAE
```bash
python train_vqvae.py  \
--output_dir=${output_root} \
--model_name=vqvae_decathlon \
--training_ids=${data_root}/data_splits/Task01_BrainTumour_train.csv \
--validation_ids=${data_root}/data_splits/Task01_BrainTumour_val.csv  \
--is_grayscale=1 \
--n_epochs=300 \
--batch_size=8  \
--eval_freq=10 \
--cache_data=0  \
--vqvae_downsample_parameters=[[2,4,1,1],[2,4,1,1],[2,4,1,1],[2,4,1,1]] \
--vqvae_upsample_parameters=[[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0]] \
--vqvae_num_channels=[256,256,256,256] \
--vqvae_num_res_channels=[256,256,256,256] \
--vqvae_embedding_dim=128 \
--vqvae_num_embeddings=2048 \
--vqvae_decay=0.9  \
--vqvae_learning_rate=3e-5 \
--spatial_dimension=3 \
--image_roi=[160,160,128] \
--image_size=128
```
The code is DistributedDataParallel (DDP) compatible. To train on e.g. 2 GPUs run with
`torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 train_vqvae.py`
### Train LDM
```bash
python train_ddpm.py \
  --output_dir=${output_root} \
  --model_name=ddpm_decathlon \
  --vqvae_checkpoint=${output_root}/vqvae_decathlon/checkpoint.pth \
  --training_ids=${data_root}/data_splits/Task01_BrainTumour_train.csv \
  --validation_ids=${data_root}/data_splits/Task01_BrainTumour_val.csv  \
  --is_grayscale=1 \
  --n_epochs=12000 \
  --batch_size=6 \
  --eval_freq=25 \
  --checkpoint_every=1000 \
  --cache_data=0  \
  --prediction_type=epsilon \
  --model_type=small \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --image_roi=[160,160,128] \
  --image_size=128
```
### Reconstruct data
```bash
python reconstruct.py \
  --output_dir=${output_root} \
  --model_name=ddpm_decathlon \
  --vqvae_checkpoint=${output_root}/decathlon-vqvae-4layer/checkpoint.pth \
  --validation_ids=${data_root}/data_splits/Task01_BrainTumour_val.csv  \
  --in_ids=${data_root}/data_splits/Task01_BrainTumour_test.csv \
  --out_ids=${data_root}/data_splits/Task02_Heart_test.csv,${data_root}/data_splits/Task03_Liver_test.csv,${data_root}/data_splits/Task04_Hippocampus_test.csv,${data_root}/data_splits/Task05_Prostate_test.csv,${data_root}/data_splits/Task06_Lung_test.csv,${data_root}/data_splits/Task07_Pancreas_test.csv,${data_root}/data_splits/Task08_HepaticVessel_test.csv,${data_root}/data_splits/Task09_Spleen_test.csv\
  --is_grayscale=1 \
  --batch_size=32 \
  --cache_data=0 \
  --prediction_type=epsilon \
  --beta_schedule=scaled_linear_beta \
  --beta_start=0.0015 \
  --beta_end=0.0195 \
  --b_scale=1.0 \
  --spatial_dimension=3 \
  --image_roi=[160,160,128] \
  --image_size=128 \
  --num_inference_steps=100 \
  --inference_skip_factor=2 \
  --run_val=1 \
  --run_in=1 \
  --run_out=1
````
### Classify samples as OOD
```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=ddpm_decathlon
```
## Acknowledgements
Built with [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels) and [MONAI](https://github.com/Project-MONAI/MONAI).


## Citations
If you use this codebase, please cite
```bib
@InProceedings{Graham_2023_CVPR,
    author    = {Graham, Mark S. and Pinaya, Walter H.L. and Tudosiu, Petru-Daniel and Nachev, Parashkev and Ourselin, Sebastien and Cardoso, Jorge},
    title     = {Denoising Diffusion Models for Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2947-2956}
}
