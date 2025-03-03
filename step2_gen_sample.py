import json
import os
import datetime
import argparse
import torch
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from peft import AutoPeftModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='gemma2b')
parser.add_argument("--adapter_name", type=str, default='sft')
parser.add_argument("--dataset", type=str, default="hh-rlhf-helpful") # hh-rlhf
parser.add_argument("--eval_dataset", type=str, default="ultrafeedback") # hh-rlhf
parser.add_argument("--gpu_idx", type=int, default=0)
parser.add_argument("--n_samples", type=int, default=500)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument("--output_dir", type=str, default=None) # generated_responses/ ,  generated_summaries/
parser.add_argument("--data_class", type = str, default='test')
parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--split", type=int, default=0)
args = parser.parse_args()

config = vars(args)
torch.cuda.set_device(args.gpu_idx)
output_dir = f"{args.output_dir}/Part_{args.split}_sft_{args.adapter_name}max_len{args.max_len}_temp{args.temperature}_{args.model_name}_{args.dataset}_{args.eval_dataset}_n{args.n_samples}_dcls{args.data_class}/"
os.makedirs(output_dir, exist_ok=True)

model_list = {
    "gemma2b": "google/gemma-2b",
    "gemma7b": "google/gemma-7b",
    "llama38b": "meta-llama/Meta-Llama-3-8B",
}
MODEL_PATH = model_list[args.model_name]
# in step 1, SFT_OUT_PATH = f"SFT_{args.model_name}_{args.dataset}_lr{args.learning_rate}_epoch{args.epochs}"
ADAPTER_PATH = f'{args.output_dir}/ckpts_SFT/SFT_{args.model_name}_{args.dataset}_lr5e-05_epoch2'

if "hh-rlhf" in args.eval_dataset:
    if args.eval_dataset == "hh-rlhf-helpful":
        dataset = load_from_disk(f"local_hh_rlhf_dataset_helpful_{args.data_class}")
    elif args.eval_dataset == "hh-rlhf-harmless":
        dataset = load_from_disk(f"local_hh_rlhf_dataset_harmless_{args.data_class}")
    else:
        dataset = None
    total_len = len(dataset)
    split_len = int(total_len / 5)
    dataset = dataset.select(range(split_len * args.split, split_len * (args.split + 1)))

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = "\n\nAssistant:"
        search_term_idx = prompt_and_response.rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        return prompt_and_response[: search_term_idx + len(search_term)]
    def split_prompt_and_responses(sample):
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {"prompt": prompt} # only need the prompt
    dataset = dataset.map(split_prompt_and_responses)

elif "helpsteer" in args.eval_dataset:
    dataset = load_dataset("csv", data_files=f'data/helpsteer_{args.data_class}_prompts_cleaned.csv', split="train")
    total_len = len(dataset)
    split_len = int(total_len / 5)
    dataset = dataset.select(range(split_len * args.split, split_len * (args.split + 1)))

    def add_prefix(example):
        example['prompt'] = 'Human: ' + example['prompt'] + "\n\nAssistant:"
        return example
    dataset = dataset.map(add_prefix)

elif "ultrafeedback" in args.eval_dataset:
    dataset = load_dataset("csv", data_files=f'data/ultrafeedback_{args.data_class}_prompts_cleaned.csv', split="train")
    total_len = len(dataset)
    split_len = int(total_len / 5)
    dataset = dataset.select(range(split_len * args.split, split_len * (args.split + 1)))

    def add_prefix(example):
        example['prompt'] = 'Human: ' + example['instruction'] + "\n\nAssistant:"
        return example
    dataset = dataset.map(add_prefix)

def get_data_type():
    device = torch.cuda.current_device()
    compute_capability = torch.cuda.get_device_capability(device)
    # bf16 is supported on compute capability 8.0 and higher
    if compute_capability >= (8, 0):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16
    return data_type

sampling_params = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_len)
dtype = get_data_type()
llm = LLM(model=MODEL_PATH, gpu_memory_utilization=args.gpu_memory_utilization, swap_space=1, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, dtype=dtype, enable_lora=True, max_lora_rank=32)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

for i in tqdm(range(len(dataset))): # number of demonstrations to create
    query = dataset[i]['prompt']
    tokenized_query = tokenizer.encode(query)
    print('tokenized_query length:', len(tokenized_query))
    if len(tokenized_query) > 450:
        continue
    # Use vllm for response generation
    if args.adapter_name is None:
        batch_response = llm.generate([query for _ in range(args.n_samples)], sampling_params)
    else:
        batch_response = llm.generate([query for _ in range(args.n_samples)], sampling_params, lora_request=LoRARequest("lora_adapter_name", 1, ADAPTER_PATH))

    response_texts = []
    for j in range(args.n_samples):
        config[f'response_{j}'] = batch_response[j].outputs[0].text
        response_texts.append(query + batch_response[j].outputs[0].text)

    config['query'] = query
    config['id'] = i
    config["time_stamp"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{output_dir}config.json", "a+") as f:
        json.dump(config, f)
        f.write("\n")