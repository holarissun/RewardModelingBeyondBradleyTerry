import os
import torch
from tqdm import tqdm
import os
import json
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='gemma2b')
parser.add_argument("--adapter_name", type=str, default=None)
parser.add_argument("--dataset", type=str, default="nonsft") # hh-rlhf
parser.add_argument("--gpu_idx", type=int, default=0)
parser.add_argument("--n_samples", type=int, default=10)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--output_dir", type=str, default=None) # generated_responses/ ,  generated_summaries/
parser.add_argument("--subset", type=int, default=-1)
parser.add_argument("--data_class", type = str, default='train')
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--server_alias', type=str, default='lq')
parser.add_argument('--eval_dataset', type=str, default=None)
args = parser.parse_args()

if 'helpful' in args.dataset:
    args.eval_rm_name = 'rmmistral7b'
elif 'harmless' in args.dataset:
    args.eval_rm_name = 'rayharmless'

response_path = f"{args.output_dir}/Part_{args.split}_sft_{args.adapter_name}max_len{args.max_len}_temp{args.temperature}_{args.model_name}_{args.dataset}_{args.eval_dataset}_n{args.n_samples}_dcls{args.data_class}/"
json_file = os.path.join(response_path, 'config.json')
response_data = []
with open(json_file) as f:
    for line in f:
        config = json.loads(line)
        response_data.append(config)

if args.eval_rm_name == 'rmmistral7b':
    rm_path = 'weqweasdas/RM-Gemma-7B'
elif args.eval_rm_name == 'rayharmless':
    rm_path = 'Ray2333/gpt2-large-harmless-reward_model'
else:
    rm_path = None

def extract_model_gen_answer(answer_init_context):
    """Extract the rejected answer from the initial context of a rejected answer."""
    search_term = "\n\nHuman"
    search_term_idx = answer_init_context.find(search_term)
    if search_term_idx == -1:
        return answer_init_context.strip()
    return answer_init_context[:search_term_idx].strip()

rank_model = AutoModelForSequenceClassification.from_pretrained(
    rm_path,
    num_labels=1,
    output_attentions=False,
    return_dict_in_generate=True,
    attn_implementation="eager",
).to(f"cuda:{args.gpu_idx}")

rank_tokenizer = AutoTokenizer.from_pretrained(rm_path)
rank_tokenizer.pad_token = rank_tokenizer.eos_token
rank_model.config.pad_token_id = rank_model.config.eos_token_id

for idx in tqdm(range(len(response_data))):
    instance = response_data[idx]
    input_query = instance['query']
    trunc_response_list = []
    total_batch = args.n_samples // 10
    for batch_i in range(total_batch):
        for i in range(10):
            model_gen_answer = extract_model_gen_answer(instance[f'response_{batch_i*10 + i}'])
            decoded_batch_truncated = model_gen_answer
            trunc_response_list.append(decoded_batch_truncated)

        response_for_eval_list = []
        for i in range(10):
            response_for_eval_list.append(input_query + trunc_response_list[i])
        with torch.no_grad():
            rank_input = rank_tokenizer(response_for_eval_list,
                                        padding="max_length",
                                        max_length=512,
                                        truncation=True,
                                        return_tensors="pt",
                                    ).to(f"cuda:{args.gpu_idx}")
            rank_output = rank_model(**rank_input)
        scores = rank_output.logits
        og_reward = [score.cpu().sum() for score in scores]

        for j in range(10):
            instance[f'rm_{args.eval_rm_name}_{batch_i*10 +j}'] = og_reward[j].item()
            instance[f'trunc_response_{batch_i*10 +j}'] = trunc_response_list[j]
    with open(f"{response_path}_rm{args.eval_rm_name}_{args.split}_maxlen{args.max_len}.json", "a+") as f:
        json.dump(instance, f)
        f.write("\n")