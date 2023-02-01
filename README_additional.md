## Commands for running codebase on other datasets

### CIFAR10

```bash
python train.py \
--output_dir=${output_root} \
--model_name=cifar10 \
--training_ids=${data_root}/data_splits/CIFAR10_train.csv \
--validation_ids=${data_root}/data_splits/CIFAR10_val.csv \
--is_grayscale=0 \
--config_diffusion_file=src/configs/diffusion/diffusion_colour.yaml \
--n_epochs=300
```

```bash
python reconstruct.py \
--output_dir=${output_root} \
--model_name=cifar10 \
--validation_ids=${data_root}/data_splits/CIFAR10_val.csv \
--in_ids=${data_root}/data_splits/CIFAR10_test.csv \
--out_ids=${data_root}/data_splits/SVHN_test.csv,${data_root}/data_splits/CelebA_test.csv,${data_root}/data_splits/CIFAR10_vflip_test.csv,${data_root}/data_splits/CIFAR10_hflip_test.csv \
--is_grayscale=0 \
--config_diffusion_file=src/configs/diffusion/diffusion_colour.yaml \
--num_inference_steps=100 \
--inference_skip_factor=16 \
--run_val=1 \
--run_in=1 \
--run_out=1
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=cifar10
```

### CelebA

```bash
python train.py \
--output_dir=${output_root} \
--model_name=celeba \
--training_ids=${data_root}/data_splits/CelebA_train.csv \
--validation_ids=${data_root}/data_splits/CelebA_val.csv \
--is_grayscale=0 \
--config_diffusion_file=src/configs/diffusion/diffusion_colour.yaml \
--n_epochs=300
```

```bash
python reconstruct.py \
--output_dir=${output_root} \
--model_name=celeba \
--validation_ids=${data_root}/data_splits/CelebA_val.csv \
--in_ids=${data_root}/data_splits/CelebA_test.csv \
--out_ids=${data_root}/data_splits/SVHN_test.csv,${data_root}/data_splits/CIFAR10_test.csv,${data_root}/data_splits/CelebA_vflip_test.csv,${data_root}/data_splits/CelebA_hflip_test.csv \
--is_grayscale=0 \
--config_diffusion_file=src/configs/diffusion/diffusion_colour.yaml \
--num_inference_steps=100 \
--inference_skip_factor=16 \
--run_val=1 \
--run_in=1 \
--run_out=1
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=celeba
```
### SVHN

```bash
python train.py \
--output_dir=${output_root} \
--model_name=svhn \
--training_ids=${data_root}/data_splits/SVHN_train.csv \
--validation_ids=${data_root}/data_splits/SVHN_val.csv \
--is_grayscale=0 \
--config_diffusion_file=src/configs/diffusion/diffusion_colour.yaml \
--n_epochs=300
```

```bash
python reconstruct.py \
--output_dir=${output_root} \
--model_name=svhn \
--validation_ids=${data_root}/data_splits/SVHN_val.csv \
--in_ids=${data_root}/data_splits/SVHN_test.csv \
--out_ids=${data_root}/data_splits/CelebA_test.csv,${data_root}/data_splits/CIFAR10_test.csv,${data_root}/data_splits/SVHN_vflip_test.csv,${data_root}/data_splits/SVHN_hflip_test.csv \
--is_grayscale=0 \
--config_diffusion_file=src/configs/diffusion/diffusion_colour.yaml \
--num_inference_steps=100 \
--inference_skip_factor=16 \
--run_val=1 \
--run_in=1 \
--run_out=1
```

```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=svhn
```
