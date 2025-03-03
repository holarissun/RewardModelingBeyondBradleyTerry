import os
import torch
from tqdm import tqdm
import os
import json
from tqdm import tqdm
import numpy as np
import argparse
from typing import Dict, Optional
from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
parser = argparse.ArgumentParser()
parser.add_argument("--embed_model_name", type=str, default='gemma2b')
parser.add_argument("--dataset", type=str, default="hh-rlhf-helpful") # hh-rlhf
parser.add_argument("--output_dir", type=str, default='temp_out_dir') # generated_responses/ ,  generated_summaries/
parser.add_argument('--server_alias', type=str, default='lq')
parser.add_argument("--gen_pref_model_name", type=str, default='gemma2b')
parser.add_argument("--split", type=int, default=4)
parser.add_argument("--n_samples", type=int, default=500)
parser.add_argument("--train_test", type=str, default='test')
parser.add_argument("--max_len", type=int, default=128)
args = parser.parse_args()


if args.embed_model_name == "gemma2b":
    model_name_or_path = "google/gemma-2b",
elif args.embed_model_name == "gemma7b":
    model_name_or_path = "google/gemma-7b"
elif args.embed_model_name == "llama38b":
    model_name_or_path = "meta-llama/Meta-Llama-3-8B"
else:
    model_name_or_path = None

model_kwargs = dict(
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=1, output_attentions=False, **model_kwargs
).to("cuda")
model.config.pad_token_id = model.config.eos_token_id

folder_name = f"{args.output_dir}/"
raw_datasets = load_dataset("json", data_files=folder_name+'config.json', split='train')

embedding_dataset_total = []
process_batch_size = 50
for idx in tqdm(range(0, len(raw_datasets), process_batch_size)):
    instance = raw_datasets[idx:idx+process_batch_size]
    input_query = instance['query']
    embedding_dataset_temp = []
    for sample_idx in range(args.n_samples):
        input_response = instance[f'response_{sample_idx}']
        # tokenized_batch = tokenizer(input_response, padding="max_length", max_length=args.max_len, truncation=True)
        # decoded_batch_truncated = tokenizer.decode(tokenized_batch['input_ids'])
        inputs = tokenizer(input_query, input_response, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        embeddings_chosen = outputs.hidden_states[-1][:,-1,:].detach().cpu().numpy()
        embedding_dataset_temp.append(embeddings_chosen)
        print(embeddings_chosen.shape)
    embedding_dataset_total.append(embedding_dataset_temp)

# concate the embeddings of each sample
final_out_embedding = embedding_dataset_total[0]
for idx in range(1, len(embedding_dataset_total)):
    final_out_embedding = np.concatenate((final_out_embedding, embedding_dataset_total[idx]), axis=1)


final_out_embedding = torch.tensor(final_out_embedding)

torch.save(final_out_embedding, args.output_dir + f"/{args.embed_model_name}_{args.gen_pref_model_name}_{args.split}_embedding_total_{args.max_len}.pt")
