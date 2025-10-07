import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA = {
    "train": ".jsonl", 
    "val":   ".jsonl"
}
OUT_DIR = ""
USE_QLORA = True            
MAX_LEN = 1024             
SEED = 42

torch.manual_seed(SEED)

ds = load_dataset("json", data_files=DATA)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.eos_token

def build_prompt(inst: str) -> str:
    return f"Instruction: {inst}\n\nAnswer:"

def tok_fn(batch):
    prompts = [build_prompt(x) for x in batch["instruction"]]
    outs    = [o.strip() for o in batch["output"]]

    full_texts = [p + " " + o for p, o in zip(prompts, outs)]

    enc = tok(
        full_texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        add_special_tokens=True,
    )

    prompt_enc = tok(
        prompts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

    labels = []
    for i, p_len in enumerate(prompt_lens):
        full_ids = enc["input_ids"][i]
        lab = [-100] * p_len + full_ids[p_len:]
        lab = lab[:len(full_ids)]
        if len(lab) < len(full_ids):
            lab += [-100] * (len(full_ids) - len(lab))
        labels.append(lab)

    enc["labels"] = labels
    return enc

remove_cols = ds["train"].column_names
ds = ds.map(tok_fn, batched=True, remove_columns=remove_cols, desc="Tokenizing")


if USE_QLORA:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj","o_proj"],  
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# training
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=500,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
    disable_tqdm=False,       
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    data_collator=default_data_collator,  # inputs & labels already aligned to MAX_LEN
)

trainer.train()
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print(f"Adapter saved to: {OUT_DIR}")
