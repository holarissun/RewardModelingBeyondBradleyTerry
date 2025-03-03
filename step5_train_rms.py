import os
import torch
from tqdm import tqdm
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
import argparse
from typing import Dict, Optional
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
import lightgbm as lgb
from networks import MLP, forward_siamese, train_model, save_model

parser = argparse.ArgumentParser()
parser.add_argument("--embed_model_name", type=str, default='gemma2b')
parser.add_argument("--task", type=str, default="helpful") # train or test
parser.add_argument("--sft_obj", type=str, default="gpt4") # hh-rlhf
parser.add_argument("--output_dir", type=str, default='temp_out_dir') # generated_responses/ ,  generated_summaries/
parser.add_argument('--server_alias', type=str, default='lq')
parser.add_argument("--gen_pref_model_name", type=str, default='gemma2b')
parser.add_argument("--ensemble_number", type=int, default=10)
parser.add_argument("--rm_objective",type=str,choices=['clf','bt'], default='bt')
parser.add_argument("--consider_first_n", type=int, default=2)
parser.add_argument("--n_sample", type=int, default=10000)
parser.add_argument("--n_pairs", type=int, default=1)
parser.add_argument("--training_epochs",type=int,default=30)
parser.add_argument("--learning_rate",type=float,default=0.001)
parser.add_argument("--replacement", type=str, default='replacement_false', choices=['replacement_true', 'replacement_false'])
parser.add_argument("--seed", type=int, default=6)
parser.add_argument("--annotation_quality", type=float, default=10)
args = parser.parse_args()
args.sft_obj = '' if args.sft_obj == 'none' else '-gpt4'
args_replacement = True if args.replacement == 'replacement_true' else False

np.random.seed(args.seed)
args.rm_name = 'rmmistral7b' if 'helpful' in args.task else 'rayharmless'

# load the embedding tensors
embedding_dataset = []
scores_dataset = []
query_dataset = []
for i in range(10):
    embedding_dataset.append(torch.load(f"/mnt/bn/hsunvolume-{args.server_alias}/Step4_2_reorg_tensors_all/embed_model_name_{args.embed_model_name}_dataset_orig_hh-rlhf-{args.task}{args.sft_obj}_gen_pref_model_name_{args.gen_pref_model_name}_golden_rm_name_{args.rm_name}_train_test_train_server_alias_lq_embedding_only_rank_{i}.pt").numpy())
    if 'helpful' in args.task:
        dataset = load_dataset("json", data_files=f"data_10split/rmmistral7b_{args.gen_pref_model_name}_train_hh-rlhf-{args.task}{args.sft_obj}_database_split{i}.json", split='train')
        scores_dataset.append(dataset['scores_sorted'])
        query_dataset.append(dataset['query'])
    elif 'harmless' in args.task:
        dataset = load_dataset("json", data_files=f"data_10split/rayharmless_{args.gen_pref_model_name}_train_hh-rlhf-{args.task}{args.sft_obj}_database_split{i}.json", split='train')
        scores_dataset.append(dataset['scores_sorted'])
        query_dataset.append(dataset['query'])

for i in range(10):
    print(np.shape(embedding_dataset[i]))


# make some sanity check on the generations:
print('shape of embedding should match', np.shape(embedding_dataset))
print('shape of scores should match', np.shape(scores_dataset))
print('shape of query should match', np.shape(query_dataset))
assert np.shape(embedding_dataset)[1] == np.shape(scores_dataset)[1] == np.shape(query_dataset)[1]

positive_sample = []
negative_sample = []
idx_set_list = []
for i in range(args.n_sample):
    if args.consider_first_n > 0:
        idx_set_list.append(np.random.choice(10, args.consider_first_n, replace=False))
    elif args.consider_first_n == -2: # lacking diversity ablation study
        idx_set_list.append([4,5])
    elif args.consider_first_n == -1: # sufficient diversity ablation study
        idx_set_list.append([0,9])

p_list = [] # track statistics
reward_diff_list = []
rew_scale_factor = 6.0 if 'helpful' in args.task else 1.0
for i in range(args.n_sample):
    idx_set_of_prompt_i = np.random.choice(idx_set_list[i], args.n_pairs, replace=args_replacement)
    idx_j_set = np.random.choice(args.n_sample, args.n_pairs, replace=False) # select the second prompt randomly

    for j, idx_j in enumerate(idx_j_set):
        temp_selected_prompt_idx = np.random.choice(idx_set_list[idx_j], 1, replace=False).item()
        if args.annotation_quality < 0:
            if scores_dataset[temp_selected_prompt_idx][idx_j] > scores_dataset[idx_set_of_prompt_i[j]][i]:
                positive_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
                negative_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
            else:
                positive_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
                negative_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
        else: # use bradley-terry model to generate noisy labels
            delta_reward = scores_dataset[temp_selected_prompt_idx][idx_j] - scores_dataset[idx_set_of_prompt_i[j]][i]
            delta_reward = delta_reward / rew_scale_factor
            prob = 1 / (1 + np.exp(-delta_reward * args.annotation_quality))
            if np.random.rand() < prob:
                positive_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
                negative_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
            else:
                positive_sample.append(embedding_dataset[idx_set_of_prompt_i[j]][i])
                negative_sample.append(embedding_dataset[temp_selected_prompt_idx][idx_j])
            p_list.append(prob)
            reward_diff_list.append(delta_reward)

print('mean of p:', np.mean(p_list), 'stats', np.abs(np.asarray(p_list) - 0.5).mean())
print('mean of reward_diff:', np.mean(reward_diff_list), 'stats', np.abs(reward_diff_list).mean())
# save the p_list and reward_diff_list
p_list_fn = f'/plist_XPrompt_mlp_{args.rm_objective}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.npy'
np.save(f'{args.output_dir}'+p_list_fn, p_list)
reward_diff_list_fn = f'/reward_diff_list_XPrompt_mlp_{args.rm_objective}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.npy'
np.save(f'{args.output_dir}'+reward_diff_list_fn, reward_diff_list)
print(f"p_list saved as: {args.output_dir + p_list_fn}")
print(f"reward_diff_list saved as: {args.output_dir + reward_diff_list_fn}")


positive_sample = torch.tensor(positive_sample)
negative_sample = torch.tensor(negative_sample)
positive_label = torch.ones(positive_sample.size(0))
negative_label = torch.zeros(negative_sample.size(0))
embedding_dataset = torch.cat([positive_sample, negative_sample], dim=0)
embedding_labels = torch.cat([positive_label, negative_label], dim=0)

siamese_dataset_positive = torch.cat([positive_sample.unsqueeze(1), negative_sample.unsqueeze(1)], dim=1)
siamese_dataset_negative = torch.cat([negative_sample.unsqueeze(1), positive_sample.unsqueeze(1)], dim=1)
siamese_labels_positive = torch.ones(positive_sample.size(0))
siamese_labels_negative = torch.zeros(negative_sample.size(0))
# create the second dimension and concate in the second dimension (additional dimension)

siamese_dataset = torch.cat([siamese_dataset_positive, siamese_dataset_negative], dim=0)
siamese_labels = torch.cat([siamese_labels_positive, siamese_labels_negative], dim=0)

indices = torch.randperm(embedding_dataset.size(0))
embedding_dataset = embedding_dataset[indices]
embedding_labels = embedding_labels[indices]

indices_siamese = torch.randperm(siamese_dataset.size(0))
siamese_dataset = siamese_dataset[indices_siamese]
siamese_labels = siamese_labels[indices_siamese]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.rm_objective == 'clf':
    for model_i in range(args.ensemble_number):
        torch.manual_seed(args.seed * 42 + model_i)
        np.random.seed(args.seed * 42 + model_i)
        # prepare dataset
        X_train, X_test, y_train, y_test = train_test_split(embedding_dataset.numpy(), embedding_labels.numpy(), test_size=0.2, random_state=42 + model_i)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)  # Reshape for binary output
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)


        input_dim = X_train_tensor.shape[-1]
        model = MLP(input_dim)

        # train and save model
        train_model(model, device, 'clf', X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=args.training_epochs, lr=args.learning_rate, batch_size=10240)
        save_model(model, args.output_dir + f"/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt")
        print(f"Model saved as: {args.output_dir + f'/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt'}")
elif args.rm_objective == 'bt':
    for model_i in range(args.ensemble_number):
        torch.manual_seed(args.seed * 42 + model_i)
        np.random.seed(args.seed * 42 + model_i)
        X_train, X_test, y_train, y_test = train_test_split(siamese_dataset.numpy(), siamese_labels.numpy(), test_size=0.2, random_state=42 + model_i)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        input_dim = X_train_tensor.shape[-1]
        model = MLP(input_dim)

        # train and save model
        train_model(model, device, 'siamese', X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=args.training_epochs, lr=args.learning_rate, batch_size=10240)
        save_model(model, args.output_dir + f"/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt")
        print(f"Model saved as: {args.output_dir + f'/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt'}")