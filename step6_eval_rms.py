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
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--embed_model_name", type=str, default='gemma2b')
parser.add_argument("--task", type=str, default="helpful") # pref
parser.add_argument("--sft_obj", type=str, default='-gpt4') # hl
parser.add_argument("--output_dir", type=str, default='temp_out_dir') # generated_responses/ ,  generated_summaries/
parser.add_argument('--server_alias', type=str, default='lq')
parser.add_argument("--gen_pref_model_name", type=str, default='gemma2b')
parser.add_argument("--ensemble_number", type=int, default=3)
parser.add_argument("--rm_objective",type=str,choices=['clf','bt'], default='bt')
parser.add_argument("--consider_first_n", type=int, default=-2)
parser.add_argument("--n_sample", type=int, default=10000)
parser.add_argument("--n_pairs", type=int, default=1)
parser.add_argument("--training_epochs",type=int,default=30)
parser.add_argument("--learning_rate",type=float,default=0.001)
parser.add_argument("--normal_or_xprompt",type=str,default='xprompt',choices=['normal','xprompt'])
parser.add_argument("--replacement", type=str, default='replacement_false', choices=['replacement_true', 'replacement_false'])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_len", type=int, default=-1)
parser.add_argument("--annotation_quality", type=float, default=10)
args = parser.parse_args()
args.sft_obj_name = 'none' if args.sft_obj == '' else 'gpt4'
args.sft_obj = '' if args.sft_obj == 'none' else '-gpt4'
args_annotation_quality = {10.0: '10_0',
                           5.0: '5_0',
                           3.0: '3_0',
                           1.0: '1_0',
                           0.7: '0_7',
                           0.5: '0_5',
                           0.3: '0_3',
                           0.1: '0_1',
                           0.01: '0_01',
                           0.001: '0_001'}
args_replacement = True if args.replacement == 'replacement_true' else False
rm_name = 'rmmistral7b' if args.task == 'helpful' else 'rayharmless'

max_len_list = [128]
for max_len in max_len_list:
    args.max_len = max_len
    results = {}
    for split in range(5):
        results[f'split{split}'] = {}
        for test_gen_pref_model_name in [args.gen_pref_model_name]: #
            if args.embed_model_name == 'gemma2b':
                embedding_dataset = torch.load(f"/mnt/bn/hsunvolume-lq/Step4_2_testall/EXPembed_model_name_{args.embed_model_name}_dataset_orig_hh-rlhf-{args.task}{args.sft_obj}_gen_pref_model_name_{test_gen_pref_model_name}_n_samples_500_max_len_128_split_{split}_server_alias_lq/{args.embed_model_name}_{test_gen_pref_model_name}_hh-rlhf-{args.task}{args.sft_obj}_{split}_testembedding_total_128.pt")
            elif args.embed_model_name == 'gemma7b':
                embedding_dataset = torch.load(f"/mnt/bn/hsunvolume-lq/Step4_2_testall_7b/EXPembed_model_name_{args.embed_model_name}_dataset_orig_hh-rlhf-{args.task}{args.sft_obj}_gen_pref_model_name_{test_gen_pref_model_name}_n_samples_500_max_len_128_split_{split}_server_alias_lq/{args.embed_model_name}_{test_gen_pref_model_name}_hh-rlhf-{args.task}{args.sft_obj}_{split}_testembedding_total_128.pt")
            else:
                embedding_dataset = None
                raise ValueError(f"Invalid embed_model_name: {args.embed_model_name}")
            file_name = f'/mnt/bn/hsunvolume-lq/Step3_Annotate_Test_RPT/EXPmodel_name_{test_gen_pref_model_name}_dataset_hh-rlhf-{args.task}{args.sft_obj}_n_samples_500_split_{split}_data_class_test_server_alias_yg_rm{rm_name}_{split}_maxlen128.json'
            reward_list=[]
            id_set = []
            with open(file_name) as f:
                for line in f:
                    sub_reward_list = []
                    out_data = json.loads(line)
                    if out_data['id'] in id_set:
                        continue
                    id_set.append(out_data['id'])

                    for idx in range(500):
                        sub_reward_list.append(out_data[f'rm_{rm_name}_{idx}'])
                    reward_list.append(sub_reward_list)
            assert len(reward_list) == embedding_dataset.shape[1]

            # load lgb models

            model_predictions = []

            for model_i in range(args.ensemble_number-1,args.ensemble_number):
                # load mlp model from the checkpoint
                model = MLP(embedding_dataset.shape[2])
                if args.normal_or_xprompt == 'xprompt':
                    pre_fix = f"/mnt/bn/hsunvolume-{args.server_alias}/Step5_CNT_ML_ALLCASE_ET3_beta/EXPembed_model_name_{args.embed_model_name}_task_{args.task}_sft_obj_{args.sft_obj_name}_rm_objective_{args.rm_objective}_annotation_quality_{args_annotation_quality[args.annotation_quality]}_replacement_{args.replacement}_gen_pref_model_name_{test_gen_pref_model_name}_ensemble_number_{args.ensemble_number}_server_alias_{args.server_alias}"
                    post_fix = f"/XPrompt_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_replacement_{args_replacement}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt"
                elif args.normal_or_xprompt == 'normal':
                    pre_fix = f"/mnt/bn/hsunvolume-{args.server_alias}/Step5_CNT_ML_ALLCASE_ET3_beta/EXPembed_model_name_{args.embed_model_name}_task_{args.task}_sft_obj_{args.sft_obj_name}_rm_objective_{args.rm_objective}_annotation_quality_{args_annotation_quality[args.annotation_quality]}_gen_pref_model_name_{test_gen_pref_model_name}_ensemble_number_{args.ensemble_number}_server_alias_{args.server_alias}"
                    post_fix = f"/POS_mlp_{args.rm_objective}_{model_i}_seed{args.seed}_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_epoch{args.training_epochs}_lr{args.learning_rate}.ckpt"
                else:
                    pre_fix = post_fix = None
                    raise ValueError(f"Invalid normal_or_xprompt: {args.normal_or_xprompt}")
                model.load_state_dict(torch.load(pre_fix + post_fix))
                model.eval()
                model_i_all_preds = []
                for instance_i in range(embedding_dataset.shape[1]):
                    model_i_all_preds.append(model(torch.tensor(embedding_dataset[:, instance_i])).detach().numpy())
                model_predictions.append(model_i_all_preds)
            model_predictions = np.array(model_predictions)

            model_predictions_mean = model_predictions.mean(0)
            model_predictions_min = model_predictions.min(0)
            model_predictions_first = model_predictions[0]

            # calculate binary accuracy
            quantile_list_mean = []
            quantile_list_min = []
            quantile_list_first = []
            acc_list_mean = []
            acc_list_min = []
            acc_list_first = []
            for instance_i in range(embedding_dataset.shape[1]):
                max_idx_mean = np.argmax(model_predictions_mean[instance_i])
                max_idx_min = np.argmax(model_predictions_min[instance_i])
                max_idx_first = np.argmax(model_predictions_first[instance_i])

                real_value_mean = reward_list[instance_i][max_idx_mean]
                real_value_min = reward_list[instance_i][max_idx_min]
                real_value_first = reward_list[instance_i][max_idx_first]

                # calculate the quantile of the selected max in the reward list
                quantile_mean = np.sum(np.asarray(reward_list[instance_i]) <= real_value_mean) / len(reward_list[instance_i])
                quantile_min = np.sum(np.asarray(reward_list[instance_i]) <= real_value_min) / len(reward_list[instance_i])
                quantile_first = np.sum(np.asarray(reward_list[instance_i]) <= real_value_first) / len(reward_list[instance_i])

                quantile_list_mean.append(quantile_mean)
                quantile_list_min.append(quantile_min)
                quantile_list_first.append(quantile_first)

                if quantile_mean <= 0.5:
                    acc_list_mean.append(1)
                else:
                    acc_list_mean.append(0)

                if quantile_min <= 0.5:
                    acc_list_min.append(1)
                else:
                    acc_list_min.append(0)

                if quantile_first <= 0.5:
                    acc_list_first.append(1)
                else:
                    acc_list_first.append(0)

            print(f"Mean quantile of selected max using MEAN ESB {np.mean(quantile_list_mean)}")
            print(f"Mean quantile of selected max using MIN ESB {np.mean(quantile_list_min)}")
            print(f"Mean quantile of selected max using FIRST ESB {np.mean(quantile_list_first)}")
            print(f"Mean binary accuracy using MEAN ESB {np.mean(acc_list_mean)}")
            print(f"Mean binary accuracy using MIN ESB {np.mean(acc_list_min)}")
            print(f"Mean binary accuracy using FIRST ESB {np.mean(acc_list_first)}")

            quantile_list_mean = np.array(quantile_list_mean)
            quantile_list_min = np.array(quantile_list_min)
            quantile_list_first = np.array(quantile_list_first)
            acc_list_mean = np.array(acc_list_mean)
            acc_list_min = np.array(acc_list_min)
            acc_list_first = np.array(acc_list_first)


            # record the quantile of selected max

            # overall spearman correlation between predictions and reward list
            correlations = []
            for instance_i in range(embedding_dataset.shape[1]):
                # predicted_ranks = np.argsort(np.argsort())+1
                correlations.append(spearmanr(model_predictions_mean[instance_i], reward_list[instance_i]).correlation)
            correlations_mean = np.array(correlations)
            print(f"Mean Spearman correlation using MEAN ESB {np.mean(correlations_mean[~np.isnan(correlations_mean)])}")

            correlations = []
            for instance_i in range(embedding_dataset.shape[1]):
                # predicted_ranks = np.argsort(np.argsort())+1
                correlations.append(spearmanr(model_predictions_min[instance_i], reward_list[instance_i]).correlation)
            correlations_min = np.array(correlations)
            print(f"Mean Spearman correlation using MIN ESB {np.mean(correlations_min[~np.isnan(correlations_min)])}")

            correlations = []
            for instance_i in range(embedding_dataset.shape[1]):
                # predicted_ranks = np.argsort(np.argsort())+1
                correlations.append(spearmanr(model_predictions_first[instance_i], reward_list[instance_i]).correlation)
            correlations_first = np.array(correlations)
            print(f"Mean Spearman correlation using FIRST ESB {np.mean(correlations_first[~np.isnan(correlations_first)])}")

            # evaluate best of N:
            reward_mean_list = []
            reward_min_list = []
            reward_first_list = []
            for repeat_exp in range(100):
                repeat_mean_list = []
                repeat_min_list = []
                repeat_first_list = []
                np.random.seed(repeat_exp)
                sampled_idx_all_exp = np.random.choice(500, 500, replace=False)
                for N_best in [2, 5, 10, 30, 50, 100, 300, 500]:
                    sampled_idx = sampled_idx_all_exp[:N_best]
                    reduced_predictions = model_predictions[:, :, sampled_idx]
                    effect_reward_list = np.asarray(reward_list)[:, sampled_idx]
                    reduced_predictions_mean = reduced_predictions.mean(0)
                    reduced_predictions_min = reduced_predictions.min(0)
                    reduced_predictions_first = reduced_predictions[0]

                    # select the best of N and evaluate its reward

                    argmax_mean = np.argmax(reduced_predictions_mean, axis=1)
                    argmax_min = np.argmax(reduced_predictions_min, axis=1)
                    argmax_first = np.argmax(reduced_predictions_first, axis=1)

                    # calculate the reward
                    reward_mean = np.array([effect_reward_list[i][argmax_mean[i]] for i in range(embedding_dataset.shape[1])])
                    reward_min = np.array([effect_reward_list[i][argmax_min[i]] for i in range(embedding_dataset.shape[1])])
                    reward_first = np.array([effect_reward_list[i][argmax_first[i]] for i in range(embedding_dataset.shape[1])])

                    repeat_mean_list.append(reward_mean.mean())
                    repeat_min_list.append(reward_min.mean())
                    repeat_first_list.append(reward_first.mean())

                reward_mean_list.append(repeat_mean_list)
                reward_min_list.append(repeat_min_list)
                reward_first_list.append(repeat_first_list)

            print(f"Mean reward using MEAN ESB {reward_mean_list}")
            print(f"Mean reward using MIN ESB {reward_min_list}")
            print(f"Mean reward using FIRST ESB {reward_first_list}")

            reward_mean_list = np.array(reward_mean_list)
            reward_min_list = np.array(reward_min_list)
            reward_first_list = np.array(reward_first_list)
            import matplotlib.pyplot as plt
            plt.plot(reward_mean_list.mean(0), label = 'mean')
            plt.fill_between(range(8), reward_mean_list.mean(0)-reward_mean_list.std(0), reward_mean_list.mean(0)+reward_mean_list.std(0), alpha=0.3)
            plt.plot(reward_min_list.mean(0), label = 'min')
            plt.fill_between(range(8), reward_min_list.mean(0)-reward_min_list.std(0), reward_min_list.mean(0)+reward_min_list.std(0), alpha=0.3)
            plt.plot(reward_first_list.mean(0), label = 'none')
            plt.fill_between(range(8), reward_first_list.mean(0)-reward_first_list.std(0), reward_first_list.mean(0)+reward_first_list.std(0), alpha=0.3)

            plt.legend()
            plt.title(f'{test_gen_pref_model_name}')
            plt.show()

            results[f'split{split}'][test_gen_pref_model_name] = {
                'correlations_mean': np.mean(correlations_mean[~np.isnan(correlations_mean)]),
                'correlations_min': np.mean(correlations_min[~np.isnan(correlations_min)]),
                'correlations_first': np.mean(correlations_first[~np.isnan(correlations_first)]),
                'reward_mean_list': reward_mean_list.tolist(),
                'reward_min_list': reward_min_list.tolist(),
                'reward_first_list': reward_first_list.tolist(),
                'quantile_list_mean': quantile_list_mean.tolist(),
                'quantile_list_min': quantile_list_min.tolist(),
                'quantile_list_first': quantile_list_first.tolist(),
                'acc_list_mean': acc_list_mean.tolist(),
                'acc_list_min': acc_list_min.tolist(),
                'acc_list_first': acc_list_first.tolist()
            }

    # save results
    with open(f"{args.output_dir}/MLP_x_or_n_{args.normal_or_xprompt}_{args.rm_objective}_epoch{args.training_epochs}_lr{args.learning_rate}_XPrompt-RESULTS_firstn_{args.consider_first_n}_n_{args.n_sample}_pair_{args.n_pairs}_seed{args.seed}_BON-{args.max_len}_replacement_{args_replacement}.json", 'w') as f:
        json.dump(results, f)



