#!/bin/bash
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=aal
#################
#number of GPUs
#SBATCH --gres=gpu:l40s:6
#SBATCH --cpus-per-task=4
#SBATCH --account=aal
#################
#set a job name
#SBATCH --job-name="original_v1.6_llava"
#################
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=/sailhome/sunfanyun/slurm_out/%x.%j.out
#################
# a file for errors from the job
#SBATCH --error=/sailhome/sunfanyun/slurm_out/%x.%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm, 144
#SBATCH --time=13-23:59:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM)
##SBATCH -p dev
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=32G
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
##SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
##SBATCH --mail-user=fanyun@stanford.edu
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
#now run normal bash commands
#python your_command.py
#sh /viscam/u/sunfanyun/GenLayout/scripts/train_data_preprocessing.sh $dataset
export HOME=/viscam/projects/GenLayout
source /viscam/projects/SceneAug/miniconda3/etc/profile.d/conda.sh
conda activate llava
echo "activated"

model_name=llava-v1.6-mistral-7b 
version=constraint_vlm_v0

working_directory=/viscam/projects/GenLayout/GenLayout_sun/third_party/LLaVa-1.6-ft/
output_dir=$working_directory/checkpoints/$model_name-$version-finetune_task_lora
data_path=/viscam/projects/GenLayout/GenLayout_sun/data/3dfront_for_vlm_all_v0_cleaned.json
cd $working_directory

deepspeed --master_port 29700 llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/$model_name \
    --version mistral_instruct \
    --data_path $data_path \
    --image_folder / \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bf16 False \
    --fp16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \

echo "Done"
exit 0
