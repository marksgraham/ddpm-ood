#  Denoising Diffusion Models for Out-of-Distribution Detection

This repo does provides code for the paper 'Denoising Diffusion Models for Out-of-Distribution Detection'.

It is based on the excellent [Latent Diffusion repository](https://github.com/CompVis/latent-diffusion) 
(although the models in this work are vanilla diffusion models) - many thanks!
## Running the code

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.6.9) and then install the required packages:

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

### Download and process datasets
```bash
python get_datasets.py --data_root=${data_root}
```
N.B. If the error "The daily quota of the file img_align_celeba.zip is exceeded and it can't be downloaded" is thrown,
you need to download these files manually from the GDrive and place them in ${data_root}/CelebA/raw/, 
[see here](https://github.com/pytorch/vision/issues/1920#issuecomment-852237902). You can then run

```bash
python get_datasets.py --data_root=${data_root} --download_celeba=False
```
### Train models
Examples here use FashionMNIST as the in-distribution dataset. Commands for other datasets are given 
in [README_additional.md](README_additional.md).

```bash
python train.py \
--output_dir=${output_root} \
--model_name=fashionmnist \
--training_ids=${data_root}/data_splits/FashionMNIST_train.csv \
--validation_ids=${data_root}/data_splits/FashionMNIST_val.csv \
--is_grayscale=1 \
--config_diffusion_file=src/configs/diffusion/diffusion_grayscale.yaml \
--n_epochs=300
```
You can track experiments in tensorboard
```bash
tensorboard --logdir=${output_root}
```

### Reconstruct data

```bash
python reconstruct.py \
--output_dir=${output_root} \
--model_name=fashionmnist \
--validation_ids=${data_root}/data_splits/FashionMNIST_val.csv \
--in_ids=${data_root}/data_splits/FashionMNIST_test.csv \
--out_ids=${data_root}/data_splits/MNIST_test.csv,${data_root}/data_splits/MNIST_vflip_test.csv,${data_root}/data_splits/MNIST_hflip_test.csv \
--is_grayscale=1 \
--config_diffusion_file=src/configs/diffusion/diffusion_grayscale.yaml \
--num_inference_steps=100 \
--inference_skip_factor=16 \
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


# 