import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import sys
from tqdm import tqdm
# Path to your fine-tuned model

#model = "1.6"
#version = "v2"
model = str(sys.argv[1])
version = str(sys.argv[2])

if model == "1.5":
    fine_tuned_model_path = f"/svl/u/sunfanyun/GenLayout/third_party/LLaVa-1.6-ft/checkpoints/llava-v1.5-7b-task-lora_{version}"
    model_base = "liuhaotian/llava-v1.5-7b"
    #model_base = "liuhaotian/llava-v1.5-13b"
else:
    fine_tuned_model_path = f"/svl/u/sunfanyun/GenLayout/third_party/LLaVa-1.6-ft/checkpoints/llava-v1.6-mistral-7b-llava-lora-mistral_{version}"
    model_base = "liuhaotian/llava-v1.6-mistral-7b"
    #model_base = "liuhaotian/llava-v1.6-34b"

model_name =  get_model_name_from_path(fine_tuned_model_path)

print('loading ...')
tokenizer, model, image_processor, context_len = load_pretrained_model(
    fine_tuned_model_path,
    model_base,
    model_name
)
print('pretrain model loaded')

# Evaluation setup
# load the json file /svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/all_data.json
#all_data = json.load(open("/svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/all_data_v2.json", "r"))
all_data = json.load(open("/svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/rotation_merged.json", "r"))
initial_prompt = all_data[0]["conversations"][0]["value"]
# ground_truth = all_data[i]["conversations"][1]["value"]

print(len(all_data))
correct_cnt = 0
total_cnt = 0
for i in tqdm(range(1000)):
    prompt = initial_prompt #all_data[i]["conversations"][0]["value"]
    ground_truth = all_data[i]["conversations"][1]["value"]
    prompt = prompt[8:]
    image_file = all_data[i]["image"]
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
    assert type(output) == str
    #print('====================================')
    #print(ground_truth)

    correct_cnt += int(str(output) == str(ground_truth))
    total_cnt += 1

print(f"Accuracy: {correct_cnt / total_cnt}")
