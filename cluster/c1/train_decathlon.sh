# vq-vae
ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.4.norm \
--commandline "torchrun \
--nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
/mount/ddpm-ood/train_vqvae.py  \
--output_dir=/mount/output/  \
--model_name=vqvae_decathlon_task01_3layer_128  \
--training_ids=/mount/data/data_splits/Task01_BrainTumour_train.csv  \
--validation_ids=/mount/data/data_splits/Task01_BrainTumour_val.csv  \
--is_grayscale=1 \
--n_epochs=500 \
--batch_size=8  \
--eval_freq=10 \
--cache_data=1  \
--image_size=128 \
--vqvae_downsample_parameters=[[2,4,1,1],[2,4,1,1],[2,4,1,1]] \
--vqvae_upsample_parameters=[[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0]] \
--vqvae_num_channels=[128,128,128] \
--vqvae_num_res_channels=[128,128,128] \
--vqvae_embedding_dim=128 \
--spatial_dimension=3" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.1" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

# LDM
ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
            --instance dgxa100.80g.8.norm \
            --result /mandatory_results \
            --image "r5nte7msx1tj/amigo/ddp-ood:v0.1.1" \
            --org r5nte7msx1tj --team amigo \
            --workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
            --order 50 \
            --commandline "torchrun \
            --nproc_per_node=8 \
            --nnodes=1 \
            --node_rank=0 \
            /mount/ddpm-ood/train_ddpm.py \
            --output_dir=/mount/output/ \
            --model_name=ddpm_decathlon_task01_3layer_128 \
            --vqvae_checkpoint=/mount/output/vqvae_decathlon_task01_3layer_128/checkpoint_400.pth \
            --training_ids=/mount/data/data_splits/Task01_BrainTumour_train.csv \
            --validation_ids=/mount/data/data_splits/Task01_BrainTumour_val.csv \
            --is_grayscale=1 \
            --n_epochs=6000 \
            --batch_size=36 \
            --eval_freq=10 \
            --cache_data=1  \
            --prediction_type=epsilon \
            --model_type=small \
            --beta_schedule=scaled_linear \
            --beta_start=0.0015 \
            --beta_end=0.0195 \
            --b_scale=1.0 \
            --image_size=128 \
            --spatial_dimension=3"
