import os
import sys
# print
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
sys.path.append("/viscam/projects/GenLayout/GenLayout_sun/third_party/LLaVa-1.6-ft/")

from llava.train.train import train
import torch.distributed as dist

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    cleanup()
    train(attn_implementation="flash_attention_2")
