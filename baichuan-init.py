import torch
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig

model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='v1.0.4')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", 
                              trust_remote_code=True, torch_dtype=torch.float16)

model.generation_config = GenerationConfig.from_pretrained(model_dir)

messages = []
messages.append({"role": "user", "content": "讲解一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)
messages.append({'role': 'assistant', 'content': response})
messages.append({"role": "user", "content": "背诵一下将进酒"})
response = model.chat(tokenizer, messages)
print(response)