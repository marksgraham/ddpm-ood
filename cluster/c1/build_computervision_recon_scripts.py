num_gpus = 8
datasets = ["FashionMNIST", "CIFAR10", "SVHN", "CelebA"]
for data in datasets:
    model = f"monaigen_{data.lower()}_scaledlinear2"
    in_ids = f"/mount/data/data_splits/{data}_train.csv"
    val_ids = f"/mount/data/data_splits/{data}_val.csv"
    is_grayscale = 1 if data == "FashionMNIST" else 0

    out_flips = [
        f"/mount/data/data_splits/{data}_hflip_test.csv",
        f"/mount/data/data_splits/{data}_vflip_test.csv",
    ]
    if data == "FashionMNIST":
        out_ids = ["/mount/data/data_splits/MNIST_test.csv"]
        out_ids.extend(out_flips)

    else:
        out_datasets = set(datasets)
        out_datasets.remove(data)
        out_datasets.remove("FashionMNIST")
        out_ids = [f"/mount/data/data_splits/{d}_test.csv" for d in out_datasets]
        out_ids.extend(out_flips)
    out_ids = ",".join(out_ids)

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
                /mount/ddpm-ood/reconstruct.py  \
                --output_dir=/mount/output/  \
                --model_name={model}  \
                --validation_ids={in_ids} \
                --in_ids={val_ids}  \
                --out_ids={out_ids}  \
                --cache_data=0 \
                --is_grayscale={is_grayscale}  \
                --num_inference_steps=100 \
                --inference_skip_factor=2 \
                --run_val=1 \
                --run_in=1 \
                --run_out=1 \
                --batch_size={int(1024/num_gpus)} \
                --first_n_val=1024 \
                --first_n=1024 \
                --prediction_type=epsilon   \
                --model_type=small \
                --beta_schedule=scaled_linear \
                --beta_start=0.0015 \
                --beta_end=0.015 \
                --b_scale=1.0"'
    print(run_script)
    print("\ndebug")
