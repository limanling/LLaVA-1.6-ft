import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import sys
from tqdm import tqdm
import numpy as np
import re

def find_number(text):
    pattern = r"\b-?\d+(\.\d+)?\b"
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return "No number found in the text."
# Path to your fine-tuned model

#model = "1.6"
#version = "v2"
model = str(sys.argv[1])
version = str(sys.argv[2])

if model == "1.5":
    fine_tuned_model_path = f"/svl/u/sunfanyun/GenLayout/third_party/LLaVa-1.6-ft/checkpoints/llava-v1.5-7b-task-lora_{version}"
    model_name =  get_model_name_from_path(fine_tuned_model_path)
    model_base = "liuhaotian/llava-v1.5-7b"
    #model_base = "liuhaotian/llava-v1.5-13b"
elif model == "1.6":
    fine_tuned_model_path = f"/svl/u/sunfanyun/GenLayout/third_party/LLaVa-1.6-ft/checkpoints/llava-v1.6-mistral-7b-llava-lora-mistral_{version}"
    model_name =  get_model_name_from_path(fine_tuned_model_path)
    model_base = "liuhaotian/llava-v1.6-mistral-7b"
    #model_base = "liuhaotian/llava-v1.6-34b"
else:
    #fine_tuned_model_path = "liuhaotian/llava-v1.5-7b"
    #model_name =  "liuhaotian/llava-v1.5-7b"
    #model_base = None

    fine_tuned_model_path = "liuhaotian/llava-v1.6-mistral-7b"
    model_name =  "liuhaotian/llava-1.6-mistral-7b"
    model_base = None


print('loading ...')
tokenizer, model, image_processor, context_len = load_pretrained_model(
    fine_tuned_model_path,
    model_base,
    model_name
)
print('pretrain model loaded')

# Evaluation setup
# load the json file /svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/all_data.json
#all_data = json.load(open("/svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/rotation_merged.json", "r"))
all_data = json.load(open(f"/svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/all_data_{version}.json", "r"))
initial_prompt = all_data[0]["conversations"][0]["value"]
# ground_truth = all_data[i]["conversations"][1]["value"]

print(len(all_data))
correct_cnt = 0
total_cnt = 0
for i in tqdm(range(1000)):
    index = np.random.randint(0, len(all_data))
    prompt = initial_prompt #all_data[i]["conversations"][0]["value"]
    ground_truth = all_data[index]["conversations"][1]["value"]
    prompt = prompt[8:]
    image_file = all_data[index]["image"]
    #if len(prompt) > 6000:
    #    continue
    # Set up evaluation arguments
    args = type('Args', (), {
            #"model_path": fine_tuned_model_path,
            #"model_base": model_base,
            "model_name": model_name,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0.,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
    })()
    # Perform evaluation with the fine-tuned model
    output = eval_model(args, tokenizer, model, image_processor)
    print(output)
    output = str(find_number(output))
    assert type(output) == str
    print('====================================')
    print(output)
    print(ground_truth)
    print('====================================')

    correct_cnt += int(str(output) == str(ground_truth))
    total_cnt += 1

    print(f"Eval num {i}, Accuracy: {correct_cnt / total_cnt}")
