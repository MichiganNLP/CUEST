from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import torch, readline  
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER = "" 

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto"
)

adapter_path = Path(ADAPTER)
model = PeftModel.from_pretrained(
    model,
    str(adapter_path),     # local path, NOT a HF repo id
    is_trainable=False
)



streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

SYSTEM = "You are a CURIOUS agent focussed on understanding and creating cultural awareness. "
history = []

def build_prompt(user):
    conv = [
        {"role": "system", "content": SYSTEM},
        *history,
        {"role": "user", "content": user},
    ]
    text = ""
    for turn in conv:
        if turn["role"] == "system":
            text += f"<|system|>\n{turn['content']}\n"
        elif turn["role"] == "user":
            text += f"<|user|>\n{turn['content']}\n"
        else:
            text += f"<|assistant|>\n{turn['content']}\n"
    text += "<|assistant|>\n"
    return text

gen_cfg = dict(
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.05,
    do_sample=True,
)

while True:
    try:
        user = input("You: ")
        if not user: continue
        prompt = build_prompt(user)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print("Model:", end=" ", flush=True)
        out = model.generate(**inputs, streamer=streamer, **gen_cfg)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        history.append({"role":"user","content":user})
        history.append({"role":"assistant","content":reply.strip()})
    except KeyboardInterrupt:
        print("\nbye"); break
