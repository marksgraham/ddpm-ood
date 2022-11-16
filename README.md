#  Denoising Diffusion Models for Out-of-Distribution Detection

This repo does..... 

It is based on the excellent LDM repo - many thanks!
## Running the code

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.6.9) and then install the required packages:

```pip install -r requirements.txt```

You can also build the docker image
```bash
cd docker/
bash create_docker_image.sh
```
### Download and process datasets
```bash
python get_datasets.py --data_root='/desired/path/to/data'
```
N.B. If the error "The daily quota of the file img_align_celeba.zip is exceeded and it can't be downloaded." is thrown,
you need to download these files manually from the GDrive and place them in data_root/CelebA/raw/, see 
[here](https://github.com/pytorch/vision/issues/1920#issuecomment-852237902). You can then run
```bash
python get_datasets.py --data_root='/desired/path/to/data'--download_celebA=False
```

### Train models


```bash
python train.py \
--output_dir=/home/mark/data_drive/ddpm-ood/output \
--model_name=fashionmnist \
--training_ids=/home/mark/data_drive/ddpm-ood/data/data_splits/FashionMNIST_train.csv \
--validation_ids=/home/mark/data_drive/ddpm-ood/data/data_splits/FashionMNIST_val.csv \
--is_grayscale=1 \
--config_diffusion_file=/home/mark/projects/wellcome/diffusion/ddpm-ood/src/configs/diffusion/diffusion_grayscale.yaml \
--n_epochs=300
```

### Reconstruct data

```bash
python reconstruct.py \
--output_dir=/home/mark/data_drive/ddpm-ood/output \
--model_name=fashionmnist \
--validation_ids=/home/mark/data_drive/ddpm-ood/data/data_splits/FashionMNIST_val.csv \
--in_ids=/home/mark/data_drive/ddpm-ood/data/data_splits/FashionMNIST_test.csv \
--out_ids=/home/mark/data_drive/ddpm-ood/data/data_splits/MNIST_test.csv,/home/mark/data_drive/ddpm-ood/data/data_splits/MNIST_vflip_test.csv,/home/mark/data_drive/ddpm-ood/data/data_splits/MNIST_hflip_test.csv \
--is_grayscale=1 \
--config_diffusion_file=/home/mark/projects/wellcome/diffusion/ddpm-ood/src/configs/diffusion/diffusion_grayscale.yaml \
--num_inference_steps=100 \
--inference_skip_factor=16 \
--run_val=1 \
--run_in=1 \
--run_out=1
```
N.B. For a quicker run, you can choose to only reconstruct a subset of the validation set with e.g. `--first_n_val=1000` 
or a subset of the in/out datasets with `--first_n_val=1000`

The `--inference_skip_factor` controls the number of reconstructions performed. For easy reference, you can use this look-up to match


### Classify samples as OOD
```
python ood_detection.py \
--output_dir=/home/mark/data_drive/ddpm-ood/output \
--model_name=fashionmnist
```


# 