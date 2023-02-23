num_gpus = 4
datasets = ["FashionMNIST", "CIFAR10", "SVHN", "CelebA"]
for data in datasets:
    model = f"monaigen_{data.lower()}_scaledlinear3"
    train_ids = f"/mount/data/data_splits/{data}_train.csv"
    val_ids = f"/mount/data/data_splits/{data}_val.csv"
    is_grayscale = 1 if data == "FashionMNIST" else 0

    run_script = f'ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
                --instance dgxa100.80g.{num_gpus}.norm \
                --result /mandatory_results \
                --image "r5nte7msx1tj/amigo/ddp-ood:v0.1.1" \
                --org r5nte7msx1tj --team amigo \
                --workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
                --order 50 \
                --commandline "torchrun \
                --nproc_per_node={num_gpus} \
                --nnodes=1 \
                --node_rank=0 \
                /mount/ddpm-ood/train_ddpm.py \
                --output_dir=/mount/output/ \
                --model_name={model} \
                --training_ids={train_ids} \
                --validation_ids={val_ids} \
                --is_grayscale={is_grayscale} \
                --n_epochs=500 \
                --batch_size=256 \
                --eval_freq=10 \
                --cache_data=1  \
                --prediction_type=epsilon \
                --model_type=small \
                --beta_schedule=scaled_linear \
                --beta_start=0.0015 \
                --beta_end=0.010 \
                --b_scale=1.0" '
    print(run_script)
    print("\ndebug")
