import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Path to your fine-tuned model
fine_tuned_model_path = "/svl/u/sunfanyun/GenLayout/third_party/LLaVa-1.6-ft/checkpoints/llava-v1.5-7b-task-lora"
model_base = "liuhaotian/llava-v1.5-7b"
#model_base = "liuhaotian/llava-v1.5-13b"

#fine_tuned_model_path = "/svl/u/sunfanyun/GenLayout/third_party/LLaVa-1.6-ft/llava-lora-mistral/"
#model_base = "liuhaotian/llava-v1.6-34b"


# Evaluation setup
# load the json file /svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/all_data.json
all_data = json.load(open("/svl/u/sunfanyun/sceneVerse/preprocessed/ProcThor/all_data.json", "r"))
print(len(all_data))
for i in range(len(all_data)):
    prompt = all_data[i]["conversations"][0]["value"]
    ground_truth = all_data[i]["conversations"][1]["value"]
    prompt = prompt[8:]
    image_file = all_data[i]["image"]
    if len(prompt) > 6000:
        continue
    print(prompt)
    print(image_file)
    input()
    # Set up evaluation arguments
    args = type('Args', (), {
            "model_path": fine_tuned_model_path,
            "model_base": model_base,
            "model_name": get_model_name_from_path(fine_tuned_model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0.7,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
    })()
    # Perform evaluation with the fine-tuned model
    eval_model(args)
    print('====================================')
    print(ground_truth)
    input()

