from dataclasses import dataclass, field 
from accelerate import PartialState, infer_auto_device_map, init_empty_weights 
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, M2M100Config
from trl import ModelConfig, get_peft_config#,CPOConfig, CPOTrainer 
from utils.cpo_trainer_linear_attention import CPOTrainer
from utils.cpo_config import CPOConfig
import time
from transformers import TrainerCallback
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE,SIMPLE_SFT_CHAT_TEMPLATE
import torch
import pandas as pd
torch.cuda.empty_cache()
import wandb
from torch.optim import RMSprop
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM", 
)

base_model = AutoModelForCausalLM.from_pretrained("MBZUAI-Paris/Atlas-Chat-9B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("MBZUAI-Paris/Atlas-Chat-9B")
model = get_peft_model(base_model, peft_config)

wandb.login(key='86570a60523435fb4d496c0e63e8ae11c308bae2')
print('The training script of Linear attention using Atlas-chat')
model.to(dtype=torch.bfloat16)
cpo_config = CPOConfig(
    output_dir='Model',
    max_steps=10000,
    eval_strategy="epoch",
    max_target_length=256,
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=5,# default is 10
    per_device_eval_batch_size=5, # default is 10
    logging_steps=100,
    weight_decay=0.01,
    optim='rmsprop',
    warmup_steps=150,
    report_to='wandb',
    bf16=True,
    logging_first_step=True,
    save_total_limit=1,
    beta=0.3,
    run_name='Atlas_chat9B_Linear_attention_correct',
    cpo_alpha=0.8,
    eval_steps = 200,
    generate_during_eval =True, 
    max_completion_length=256,
    remove_unused_columns=False,
    max_length=256,
    max_prompt_length=256,
    logging_dir=output_dir+"/logs")

train = pd.read_csv('./data/train.csv')
val = pd.read_csv('./data/val.csv')
train.rename(columns={'darija': 'prompt','english': 'chosen', 'perturbation': 'rejected'}, inplace=True)
val.rename(columns={'darija': 'prompt','english': 'chosen', 'perturbation': 'rejected'}, inplace=True)
train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)
def process(row):
    try:
        source_language = "Moroccan dialect"
        target_language = "English"
        instruction = f"Translate this from [{source_language}] to [{target_language}]:\n{source_language}: {row['prompt']}\n{target_language}:"
        system = f'You are a native speaker of both [{source_language}] and [{target_language}]. You are an expert post editor of translations from [{source_language}] into [{target_language}] and a helpful assistant dedicated to improving translation quality. You will be provided with a source sentence in [{source_language}] and its translation in [{target_language}]. Your task is to carefully analyze the provided source sentence and translation, and suggest improvements to the translation. Note that you only need to generate a refined translation in [{target_language}] and do not generate anything else.'
        chat_style_prompt = [{"role": "system", "content": system}, {"role": "user", "content": instruction.strip()}]
        chat_chosen = [{"role": "system", "content": system}, {"role": "user", "content": instruction.strip()}, {"role": "assistant", "content": row["chosen"].strip()},]
        chat_rejected = [{"role": "system", "content": system}, {"role": "user", "content": instruction.strip()}, {"role": "assistant", "content": row["rejected"].strip()},]

        row["prompt"] = tokenizer.apply_chat_template(chat_style_prompt, tokenize=False, add_generation_prompt=True)
        row["chosen"] = tokenizer.apply_chat_template(chat_chosen, tokenize=False, add_generation_prompt=False)
        row["rejected"] = tokenizer.apply_chat_template(chat_rejected, tokenize=False, add_generation_prompt=False)
        return row
    except Exception as e:
        #print(f"Error processing row: {e}")
        return None
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"🧠 Trainable parameters: {trainable_params}")
    print(f"📦 Total parameters: {total_params}")
    print(f"🧪 Trainable %: {100 * trainable_params / total_params:.4f}%")

print_trainable_parameters(model)
ds_train = train.map(process, num_proc=4, batched=False)
ds_val = val.map(process, num_proc=4, batched=False)

trainer = CPOTrainer(
    model=model,
    args=cpo_config,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model(cpo_config.output_dir)
tokenizer.save_pretrained(cpo_config.output_dir)