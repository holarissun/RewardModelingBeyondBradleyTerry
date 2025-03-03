import os
from tqdm import tqdm
import os
import json
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets
output_dir = 'generated_responses' # need to change to the output directory of previous steps

eval_dataset = 'ultrafeedback'
for model_name in ['gemma2b', 'gemma7b', 'llama38b']:
    for data_cls in ['train', 'test']:
        if data_cls == 'test':
            n_samples = 500
        elif data_cls == 'train':
            n_samples = 10
        for dataset in ['hh-rlhf-helpful-gpt4']:
            print('current setups:', model_name, data_cls, dataset)
            if 'helpful' in dataset:
                reward_model = 'rmmistral7b'
            elif 'harmless' in dataset:
                reward_model = 'rayharmless'
            else:
                reward_model = None
            json_out_line = [{} for _ in range(n_samples)]
            for split in range(5):
                folder_name = f"{output_dir}/"

                json_file = folder_name + f'_rm{reward_model}_{split}_maxlen128.json'
                out_data = []
                with open(json_file) as f:
                    for line in f:
                        config = json.loads(line)
                        out_data.append(config)

                for line in tqdm(out_data):
                    for i in range(n_samples):
                        json_out_line[i]['query'] = line['query']
                    gen_scores = []
                    gen_responses_trunc = []
                    for sample_i in range(n_samples):
                        gen_scores.append(line[f'rm_{reward_model}_{sample_i}'])
                        gen_responses_trunc.append(line[f'trunc_response_{sample_i}'])
                    # find all sording scores and responses
                    gen_scores = np.array(gen_scores)
                    gen_responses_trunc = np.array(gen_responses_trunc)
                    gen_scores_sorted_idx = np.argsort(gen_scores)
                    gen_scores_sorted = gen_scores[gen_scores_sorted_idx]
                    gen_responses_trunc_sorted = gen_responses_trunc[gen_scores_sorted_idx]


                    for i in range(n_samples):
                        json_out_line[i]['scores_sorted'] = gen_scores_sorted[i]
                        json_out_line[i]['responses_sorted'] = gen_responses_trunc_sorted[i]

                        with open(f"temp_out_data/{reward_model}_{model_name}_{data_cls}_{eval_dataset}_database_split{i}.json", "a+") as f:
                            json.dump(json_out_line[i], f)
                            f.write("\n")
