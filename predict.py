import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
import random
import json
import torch
from transformers import Trainer, DataCollatorForSeq2Seq
from tqdm import tqdm
import gradio as gr

# with open("./data/test.jsonl", 'r', encoding='utf-8') as f:
#     test_data=json.load(f)

model_name = "Qwen/Qwen2-7B"
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

# ds_test = Dataset.from_list(test_data)
# tokenized_dataset_test = ds_test.map(preprocess_function, remove_columns=ds_test.column_names)



from peft import LoraConfig, TaskType

model_path="./output"
device = "cuda"
peft_config = LoraConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map={"": 0}, torch_dtype=torch.bfloat16)

from peft import get_peft_model, PeftModel

model = PeftModel.from_pretrained(model=model, model_id=model_path)

model.eval()

# 推理函数
def predict(example):
    prompt = f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512
        )
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

instruction = "扮演一个飞行员监督员，请识别下面的飞行员的话语是否在闲聊，并从['闲聊', '认真工作']来匹配目前飞行员的状态。。下面是飞行员的话语："
input_example = "下到六千三保持"
prompt_input = {"instruction": instruction, "input": f"{input_example} 飞行员的状态是：", "output": None}
pred = predict(prompt_input).strip()
print(pred)

#single batch 4min 30

# Gradio 接口
def gradio_interface(input_example):
    instruction = "扮演一个飞行员监督员，请识别下面的飞行员的话语是否在闲聊，并从['闲聊', '认真工作']来匹配目前飞行员的状态。"
    prompt_input = {"instruction": instruction, "input": f"{input_example} 飞行员的状态是：", "output": None}
    print(input_example)
    return predict(prompt_input)

# def test_function(input_text):
#     import pdb;pdb.set_trace()
#     return f"输入内容是：{input_text}"
# 创建 Gradio Web 界面
interface = gr.Interface(
    fn=gradio_interface,  # 绑定的函数
    inputs=gr.Textbox(lines=2, placeholder="请输入飞行员的话语..."),  # 输入框
    outputs=gr.Textbox(label="飞行员状态"),  # 输出框
    title="飞行员状态识别",  # 页面标题
    description="输入飞行员的话语，模型将判断当前状态是'闲聊'还是'认真工作'。"  # 描述
)

# def greet(name):
#     return "Hello " + name + "!"
 
# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
 
# demo.launch(share = True)   
# 启动 Gradio 服务

interface.queue().launch(share=True, server_port=11451)