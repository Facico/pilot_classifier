import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
import random
import json
import torch
from transformers import Trainer, DataCollatorForSeq2Seq
import random
from tqdm import tqdm

standard_txt = "./data/standard.txt"
data = []
instruction = "扮演一个飞行员监督员，请识别下面的飞行员的话语是否在闲聊，并从['闲聊', '认真工作']来匹配目前飞行员的状态。下面是飞行员的话语："
with open("./data/standard.txt", "r") as f:
    for i in tqdm(f):
        text = i.strip()
        if len(text) == 0: continue
        data.append({"instruction": instruction, "input": f"{text} 飞行员的状态是：", "output": "认真工作"})
        data.append({"instruction": instruction, "input": f"{text[:max(len(text)//5, 3)]} 飞行员的状态是：", "output": "认真工作"})
        data.append({"instruction": instruction, "input": f"{text[:max(len(text)//5*2, 3)]} 飞行员的状态是：", "output": "认真工作"})
        data.append({"instruction": instruction, "input": f"{text[:max(len(text)//5*3, 3)]} 飞行员的状态是：", "output": "认真工作"})
        data.append({"instruction": instruction, "input": f"{text[:max(len(text)//5*4, 3)]} 飞行员的状态是：", "output": "认真工作"})
data = data*2
with open("./data/other.txt", "r") as f:
    for i in tqdm(f):
        text = i.strip()
        if len(text) == 0: continue
        data.append({"instruction": instruction, "input": f"{text} 飞行员的状态是：", "output": "闲聊"})
        t = 0
        while(t<10):
            j, k = random.randint(0, len(instruction)), random.randint(0, len(instruction))
            if j > k:
                j, k = k, j
            if j+2 < k:
                data.append({"instruction": instruction, "input": f"{text[j:k]} 飞行员的状态是：", "output": "闲聊"})
            t += 1

with open("./data/other_special.txt", "r") as f:
    for i in tqdm(f):
        text = i.strip()
        if len(text) == 0: continue
        data.append({"instruction": instruction, "input": f"{text} 飞行员的状态是：", "output": "闲聊"})
        t = 0
        while(t<10):
            j, k = random.randint(0, len(instruction)), random.randint(0, len(instruction))
            if j > k:
                j, k = k, j
            if j+2 < k:
                data.append({"instruction": instruction, "input": f"{text[j:k]} 飞行员的状态是：", "output": "闲聊"})
            t += 1
random.shuffle(data)

train_data = data[:int(0.9*len(data))]
test_data = data[int(0.9*len(data)):]
with open("./data/train.jsonl", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False)
with open("./data/test.jsonl", 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False)
    
# 加载预训练模型和分词器
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

ds_train = Dataset.from_list(train_data)
tokenized_dataset_train = ds_train.map(preprocess_function, remove_columns=ds_train.column_names)
ds_test = Dataset.from_list(test_data)
tokenized_dataset_test = ds_test.map(preprocess_function, remove_columns=ds_test.column_names)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
from peft import get_peft_model

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./output/",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=100,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    save_on_each_node=True,
    gradient_checkpointing=True
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    tokenizer=tokenizer
)


# 模型训练
trainer.train()
trainer.save_model()
# 模型评估
results = trainer.evaluate()
print(results)

# # 推理函数
# def predict(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class = logits.argmax(dim=-1).item()
#     return "认真工作" if predicted_class == 0 else "闲聊"

# # 测试推理
# test_text = "请求雷达引导目视进近，跑道两两。"
# print(predict(test_text))  # 输出: 认真工作
