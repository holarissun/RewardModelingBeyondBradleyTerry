import os
import argparse
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments
from datasets import load_dataset, Dataset, load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model, PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gemma2b")
parser.add_argument("--dataset", type=str, default="hh-rlhf-helpful-gpt4")
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--save_steps", type=int, default=5000)
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()

torch.cuda.set_device(0)
os.makedirs(f'{args.output_dir}/ckpts_SFT', exist_ok=True)
model_list = {
    "gemma2b": "google/gemma-2b",
    "gemma7b": "google/gemma-7b",
    "llama38b": "meta-llama/Meta-Llama-3-8B",
}
MODEL_PATH = model_list[args.model_name]
OUT_PATH = f"SFT_{args.model_name}_{args.dataset}_lr{args.learning_rate}_epoch{args.epochs}"

if "gpt4" in args.dataset:
    # For GPT-4 demonstration datasets, determine the subtask from the dataset name.
    # e.g. "hh-rlhf-helpful-gpt4" -> subtask: "helpful"
    subtask = args.dataset.split("-")[2]
    csv_file = f"data/synthetic_hhrlhf_{subtask}_expert_gpt4.csv"
    dataset = load_dataset("csv", data_files=csv_file, split="train")

    def split_prompt_and_responses(sample):
        # Extract prompt from sample["query"] and return chosen response.
        return {
            "prompt": sample["query"][85:],
            "chosen": sample["response"],
        }
else:
    # For SFT-on-positive-sample datasets, subtask is the last part of the dataset name.
    # e.g. "hh-rlhf-helpful" -> subtask: "helpful"
    subtask = args.dataset.split("-")[-1]
    dataset = load_from_disk(f"local_hh_rlhf_dataset_{subtask}_train")

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt-response pair."""
        search_term = "\n\nAssistant:"
        search_term_idx = prompt_and_response.rfind(search_term)
        return prompt_and_response[: search_term_idx + len(search_term)]

    def split_prompt_and_responses(sample):
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt):],
        }

dataset = dataset.map(split_prompt_and_responses)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,).to(f'cuda:0')
peft_model = get_peft_model(base_model, lora_config).to(f'cuda:0')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    num_train_epochs=args.epochs,
    output_dir=f'{args.output_dir}/ckpts_SFT/'+OUT_PATH,
    learning_rate=args.learning_rate,
    save_strategy="steps",
    save_steps=args.save_steps,
    per_device_train_batch_size=1,
    logging_steps=10,
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{example['prompt'][i]} {example['chosen'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "Assistant:" # The token that indicates the start of a response, tokens may be different depending on the model
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    peft_model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
trainer.save_model(f'{args.output_dir}/ckpts_SFT/'+OUT_PATH)