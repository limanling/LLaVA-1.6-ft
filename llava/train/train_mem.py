from llava.train.train import train
import torch.distributed as dist

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    cleanup()
    train(attn_implementation="flash_attention_2")
