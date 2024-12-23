import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
import random
import json
import torch
from transformers import Trainer, DataCollatorForSeq2Seq
from tqdm import tqdm

with open("./data/test.jsonl", 'r', encoding='utf-8') as f:
    test_data=json.load(f)

model_name = "Qwen/Qwen2-7B"  # 替换为合适的Llama模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(example):
    MAX_LENGTH = 512
    prompt = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, add_special_tokens=False)
    outputs = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    input_ids = inputs["input_ids"] + outputs["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = inputs["attention_mask"] + outputs["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"]) + outputs["input_ids"] + [tokenizer.eos_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

ds_test = Dataset.from_list(test_data)
tokenized_dataset_test = ds_test.map(preprocess_function, remove_columns=ds_test.column_names)



from peft import LoraConfig, TaskType

model_path="./output"
peft_config = LoraConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

from peft import get_peft_model, PeftModel

model = PeftModel.from_pretrained(model=model, model_id=model_path)

model.eval()

# 推理函数
def predict(example):
    prompt = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512
        )
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

acc = []
for i in tqdm(test_data):
    pred = predict(i).strip()
    if i["output"] == pred:
        acc.append(1)
    else:
        acc.append(0)
        
import numpy as np
print(np.mean(acc))

