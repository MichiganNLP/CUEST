import os, re, time, random, logging
import torch
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as hf_logging
from huggingface_hub import login

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    device_map='auto',
    torch_dtype=torch.float16
)

random.seed(0)
torch.manual_seed(0)

df = pd.read_csv("hf://datasets/kellycyy/CulturalBench/CulturalBench-Hard.csv")
df.columns = [c.strip() for c in df.columns]
print(f"Loaded {len(df)} rows")
print("Sample data:")
print(df.head())
# df = df[:5]


def create_multiple_choice_questions(df):
    """
    Group the dataframe by question_idx and create proper multiple choice format
    """
    questions_data = []
    
    for question_idx, group in df.groupby('question_idx'):
        group = group.sort_values('data_idx')
        
        question_text = group.iloc[0]['prompt_question']
        country = group.iloc[0]['country']
        
        options = []
        correct_option = ""
        
        for i, (_, row) in enumerate(group.iterrows()):
            option_letter = chr(65 + i)  # Convert 0->A, 1->B, 2->C, 3->D
            option_text = row['prompt_option']
            options.append(f"{option_letter}) {option_text}")
            
            # Check if this is the correct answer
            if str(row['answer']).upper() == 'TRUE':
                correct_option = option_letter
        
        # Join options
        options_text = "\n".join(options)
        
        questions_data.append({
            'question_idx': question_idx,
            'country': country,
            'prompt_question': question_text,
            'options': options_text,
            'correct_option': correct_option,
            'original_data_indices': group['data_idx'].tolist()
        })
    
    return pd.DataFrame(questions_data)

# Create multiple choice questions
questions_df = create_multiple_choice_questions(df)
print(f"\nCreated {len(questions_df)} multiple choice questions")
print("\nSample multiple choice question:")
if len(questions_df) > 0:
    sample = questions_df.iloc[0]
    print(f"Question: {sample['prompt_question']}")
    print(f"Options:\n{sample['options']}")
    print(f"Correct: {sample['correct_option']}")


def build_multiple_choice_prompt(question: str, options: str, country: str) -> str:
    """
    Build a prompt for multiple choice questions
    """
    #  in {country}
    prompt = (
        f"You are answering a question about cultural norms and social behavior.\n\n"
        f"{question}\n\n"
        f"{options}\n\n"
        f"You may write at most 2 brief follow-up questions and 2 brief self-answers as scratch notes.\n"
        "Put every scratch note on its own line starting with '#'.\n"
        "Decide the correct option strictly from ONLY the listed Options.\n"
        f"Please respond with only the letter of your answer (A, B, C, or D). Do not include any explanation or additional text.\n\n"
        f"Answer:"
    )
    return prompt

def generate_choice(question: str, options: str, country: str, max_new_tokens: int = 10) -> str:
    prompt = build_multiple_choice_prompt(question, options, country)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # Use deterministic generation for consistency
            temperature=0.1,           # Very low temperature for focused answers
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.0,
        )

    # Decode only generated continuation
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = output[0][input_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return decoded


def parse_choice(text: str) -> str:
    """Extract A, B, C, or D from the model's response."""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    match = re.search(r'\b([ABCD])\b', text.upper())
    if match:
        return match.group(1)
    fallback_match = re.search(r'(?:OPTION\s*)?(?:\(?([ABCD])\)?)', text.upper())
    if fallback_match:
        return fallback_match.group(1)
    
    return ""


results = []

for i, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing questions"):
    try:
        question = str(row['prompt_question'])
        options = str(row['options'])
        country = str(row['country'])
        correct_option = str(row['correct_option'])
        
        raw_response = generate_choice(question, options, country)
        model_choice = parse_choice(raw_response)
        
        is_correct = model_choice == correct_option if correct_option else None
        
        if i < 3:
            print(f"\nDEBUG question {i+1}")
            print(f"Country: {country}")
            print(f"Question: {question}")
            print(f"Options:\n{options}")
            print(f"Raw response: '{raw_response}'")
            print(f"Model choice: '{model_choice}'")
            print(f"Correct option: '{correct_option}'")
            print(f"Correct: {is_correct}")

        results.append({
            "question_idx": row['question_idx'],
            "country": country,
            "prompt_question": question,
            "options": options,
            "correct_option": correct_option,
            "model_raw_response": raw_response,
            "model_choice": model_choice,
            "is_correct": is_correct,
            "original_data_indices": row['original_data_indices']
        })
        
    except Exception as e:
        print(f"Error processing question {i+1}: {e}")
        results.append({
            "question_idx": row.get('question_idx', i),
            "country": row.get('country', ''),
            "prompt_question": row.get('prompt_question', ''),
            "options": row.get('options', ''),
            "correct_option": row.get('correct_option', ''),
            "model_raw_response": f"ERROR: {e}",
            "model_choice": "",
            "is_correct": None,
            "original_data_indices": row.get('original_data_indices', [])
        })


results_df = pd.DataFrame(results)
out_path = "culbench/...csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
results_df.to_csv(out_path, index=False)
print(f"Saved results to: {out_path}")

print("\nResults Summary:")
print(f"Total questions: {len(results_df)}")
print(f"Questions with model responses: {len(results_df[results_df['model_choice'] != ''])}")
print(f"Questions with identified correct options: {len(results_df[results_df['correct_option'] != ''])}")

valid_answers = results_df[results_df['is_correct'].notna()]
if len(valid_answers) > 0:
    accuracy = valid_answers['is_correct'].sum() / len(valid_answers)
    print(f"Model accuracy: {accuracy:.2%}")

print("\nSample results:")
display_cols = ["country", "model_choice", "correct_option", "is_correct"]
print(results_df[display_cols].head().to_string(index=False))

print("\nDistribution of model choices:")
print(results_df['model_choice'].value_counts())

print("\nDistribution of correct options:")
print(results_df['correct_option'].value_counts())