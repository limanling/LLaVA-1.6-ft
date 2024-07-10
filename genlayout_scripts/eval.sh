#!/bin/bash -ex
export HOME=/viscam/projects/GenLayout
checkpoint_path=/viscam/projects/GenLayout/GenLayout_sun/third_party/LLaVa-1.6-ft/checkpoints/--finetune_task_lora
python run_eval.py $1 $checkpoint_path
