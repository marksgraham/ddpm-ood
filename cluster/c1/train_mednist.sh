ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.2.norm \
--commandline "torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
/mount/ddpm-ood/train.py  \
--output_dir=/mount/output/  \
--model_name=mednist_abdomenct  \
--training_ids=/mount/data/data_splits/AbdomenCT_train.csv  \
--validation_ids=/mount/data/data_splits/AbdomenCT_val.csv  \
--is_grayscale=1  \
--n_epochs=300   \
--batch_size=256   \
--eval_freq=1        \
--cache_data=1  \
--prediction_type=epsilon   \
--model_type='small' \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195 \
--b_scale=1.0" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.0" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.2.norm \
--commandline "torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
/mount/ddpm-ood/train.py  \
--output_dir=/mount/output/  \
--model_name=mednist_breastmri  \
--training_ids=/mount/data/data_splits/BreastMRI_train.csv  \
--validation_ids=/mount/data/data_splits/BreastMRI_val.csv  \
--is_grayscale=1  \
--n_epochs=300   \
--batch_size=256   \
--eval_freq=1        \
--cache_data=1  \
--prediction_type=epsilon   \
--model_type='small' \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195 \
--b_scale=1.0" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.0" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.2.norm \
--commandline "torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
/mount/ddpm-ood/train.py  \
--output_dir=/mount/output/  \
--model_name=mednist_chestct  \
--training_ids=/mount/data/data_splits/ChestCT_train.csv  \
--validation_ids=/mount/data/data_splits/ChestCT_val.csv  \
--is_grayscale=1  \
--n_epochs=300   \
--batch_size=256   \
--eval_freq=1        \
--cache_data=1  \
--prediction_type=epsilon   \
--model_type='small' \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195 \
--b_scale=1.0" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.0" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.2.norm \
--commandline "torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
/mount/ddpm-ood/train.py  \
--output_dir=/mount/output/  \
--model_name=mednist_cxr  \
--training_ids=/mount/data/data_splits/CXR_train.csv  \
--validation_ids=/mount/data/data_splits/CXR_val.csv  \
--is_grayscale=1  \
--n_epochs=300   \
--batch_size=256   \
--eval_freq=1        \
--cache_data=1  \
--prediction_type=epsilon   \
--model_type='small' \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195 \
--b_scale=1.0" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.0" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.2.norm \
--commandline "torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
/mount/ddpm-ood/train.py  \
--output_dir=/mount/output/  \
--model_name=mednist_hand  \
--training_ids=/mount/data/data_splits/Hand_train.csv  \
--validation_ids=/mount/data/data_splits/Hand_val.csv  \
--is_grayscale=1  \
--n_epochs=300   \
--batch_size=256   \
--eval_freq=1        \
--cache_data=1  \
--prediction_type=epsilon   \
--model_type='small' \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195 \
--b_scale=1.0" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.0" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace \
--instance dgxa100.80g.2.norm \
--commandline "torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
/mount/ddpm-ood/train.py  \
--output_dir=/mount/output/  \
--model_name=mednist_headct  \
--training_ids=/mount/data/data_splits/HeadCT_train.csv  \
--validation_ids=/mount/data/data_splits/HeadCT_val.csv  \
--is_grayscale=1  \
--n_epochs=300   \
--batch_size=256   \
--eval_freq=1        \
--cache_data=1  \
--prediction_type=epsilon   \
--model_type='small' \
--beta_schedule=scaled_linear \
--beta_start=0.0015 \
--beta_end=0.0195 \
--b_scale=1.0" \
--result /mandatory_results \
--image "r5nte7msx1tj/amigo/ddp-ood:v0.1.0" \
--org r5nte7msx1tj --team amigo \
--workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW \
--order 50

ngc batch run --name "Job-kcl-cbg1-ace-936696" --priority HIGH --preempt RUNONCE --min-timeslice 259200s --total-runtime 259200s --ace kcl-cbg1-ace                 --instance dgxa100.80g.4.norm                 --result /mandatory_results                 --image "r5nte7msx1tj/amigo/ddp-ood:v0.1.0"                 --org r5nte7msx1tj --team amigo                 --workspace UizB_55BRVKYwFtCVUHmbg:/mount:RW                 --order 50                 --commandline "torchrun                 --nproc_per_node=4                 --nnodes=1                 --node_rank=0                 /mount/ddpm-ood/reconstruct.py                  --output_dir=/mount/output/                  --model_name=mednist_abdomenct                  --in_ids=/mount/data/data_splits/AbdomenCT_val.csv                 --training_ids=/mount/data/data_splits/AbdomenCT_test.csv                  --validation_ids=/data/data_splits/Hand_test.csv,/data/data_splits/CXR_test.csv,/data/data_splits/HeadCT_test.csv,/data/data_splits/ChestCT_test.csv                  --is_grayscale=1                  --num_inference_steps=100                 --inference_skip_factor=2                 --run_val=1                 --run_in=1                 --run_out=1                 --batch_size=128                 --first_n_val=1024                 --first_n=1024                 --prediction_type=epsilon                   --model_type=small                 --beta_schedule=scaled_linear                 --beta_start=0.0015                 --beta_end=0.0195                 --b_scale=1.0"
