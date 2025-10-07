import os, re, json, random, logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.utils import logging as hf_logging

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

ADAPTERS_BY_DATASET = {
    "culbench":    "",
    "commonsense": "",
    "normad":      "",
}

SYSTEM_PROMPT = (
    "You are a culturally curious assistant. First, generate helpful follow-up questions "
    "to disambiguate the task, then (in a separate call) we will decide the final answer."
)

USE_4BIT = True
DTYPE = torch.float16
DEVICE_MAP = "auto"
SEED = 0

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

hf_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
random.seed(SEED); torch.manual_seed(SEED)

class RegexStop(StoppingCriteria):
    def __init__(self, tokenizer, pattern: str, window: int = 160, flags: int = re.IGNORECASE | re.MULTILINE):
        super().__init__()
        self.tokenizer = tokenizer
        self.regex = re.compile(pattern, flags)
        self.window = window
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        tail_ids = input_ids[0].tolist()[-self.window:]
        text_tail = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
        return bool(self.regex.search(text_tail))

def stop_on_mcq_final_answer(tok):     return RegexStop(tok, r'FINAL_ANSWER:\s*[ABCD]\b')
def stop_on_yesno_final_answer(tok):   return RegexStop(tok, r'FINAL_ANSWER:\s*(YES|NO)\b')
def stop_on_option_final_answer(tok):  return RegexStop(tok, r'FINAL_ANSWER:\s*.+')


class AdapterRunner:
    def __init__(self,
                 base_model: str,
                 adapters_by_dataset: Dict[str, str],
                 system_prompt: str = SYSTEM_PROMPT,
                 use_4bit: bool = True,
                 dtype=DTYPE,
                 device_map: str = DEVICE_MAP):
        self.base_model_name = base_model
        self.adapters_by_dataset = adapters_by_dataset
        self.system_prompt = system_prompt
        self.use_4bit = use_4bit
        self.dtype = dtype
        self.device_map = device_map

        self.base_model, self.base_tok = self._load_base()
        self.adapter_model, self.adapter_tok = self._load_base()
        self._load_all_dataset_adapters(self.adapter_model) 

    def _load_base(self):
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        ) if self.use_4bit else None

        tok = AutoTokenizer.from_pretrained(self.base_model_name, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        mdl = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=qconf,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
            torch_dtype=self.dtype
        )
        mdl.eval()
        mdl.generation_config.pad_token_id = tok.pad_token_id
        mdl.generation_config.eos_token_id = tok.eos_token_id
        return mdl, tok

    @staticmethod
    def _resolve_adapter_path(adapter_root: str) -> str:
        p = Path(adapter_root)
        if (p / "adapter_config.json").exists():
            return str(p)
        candidates = sorted(p.glob("checkpoint-*"), key=lambda x: x.stat().st_mtime, reverse=True)
        for c in candidates:
            if (c / "adapter_config.json").exists():
                return str(c)
        raise FileNotFoundError(f"adapter_config.json not found under {adapter_root}")

    def _load_all_dataset_adapters(self, mdl):
        for dataset_key, root in self.adapters_by_dataset.items():
            apath = self._resolve_adapter_path(root)
            if not isinstance(mdl, PeftModel):
                mdl = PeftModel.from_pretrained(mdl, apath, is_trainable=False, adapter_name=dataset_key)
            else:
                mdl.load_adapter(apath, adapter_name=dataset_key, is_trainable=False)
        self.adapter_model = mdl

    @staticmethod
    def _apply_chat(tok, system_prompt: str, user_text: str) -> str:
        if hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_text}\n<|assistant|>\n"

    @staticmethod
    def _generate_with(mdl, tok, system_prompt, user_text, max_new_tokens, do_sample, temperature, stopper=None):
        prompt = AdapterRunner._apply_chat(tok, system_prompt, user_text)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                do_sample=do_sample,
                temperature=temperature if do_sample else 0.0,
                top_p=0.9,
                repetition_penalty=1.02,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=stopper,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()


    def build_followup_prompt(self, question_block: str) -> str:
        return (
            "Generate 2–3 culturally relevant follow-up questions based on the prompt and its options.\n\n"
            f"{question_block}\n\n"
            "Follow-up Questions:"
        )

    def gen_followups_with_adapter(self, dataset_key: str, question_block: str) -> str:
        self.adapter_model.set_adapter(dataset_key)
        prompt = self.build_followup_prompt(question_block)
        return self._generate_with(
            self.adapter_model, self.adapter_tok,
            self.system_prompt, prompt,
            max_new_tokens=128, do_sample=False, temperature=0.0
        )

    def decide_mcq_letter(self, question: str, options_text: str, followups_block: str) -> str:
        user = (
            "First, read the follow-up questions and ANSWER THEM YOURSELF (internally; do NOT print your answers). "
            "Use those internal answers to REASON and select the SINGLE BEST OPTION.\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Follow-up Questions (context only):\n{followups_block.strip() or '(none)'}\n\n"
            "Output exactly one line:\nFINAL_ANSWER: <A|B|C|D>"
        )
        gen = self._generate_with(
            self.base_model, self.base_tok,
            self.system_prompt, user,
            max_new_tokens=48, do_sample=False, temperature=0.0,
            stopper=StoppingCriteriaList([stop_on_mcq_final_answer(self.base_tok)])
        )
        m = re.search(r'FINAL_ANSWER:\s*([ABCD])\b', gen.upper())
        if m:
            return m.group(1)

        user2 = (
            f"Question:\n{question}\n\nOptions:\n{options_text}\n\n"
            "Output exactly one line: FINAL_ANSWER: <A|B|C|D>"
        )
        gen2 = self._generate_with(
            self.base_model, self.base_tok,
            "", user2, 64, False, 0.0
        )
        m2 = re.search(r'FINAL_ANSWER:\s*([ABCD])\b', gen2.upper())
        return m2.group(1) if m2 else ""

    def decide_option_text(self, prompt_text: str, options_only_line: str, followups_block: str) -> str:
        user = (
            "First, read the follow-up questions and ANSWER THEM YOURSELF (internally; do NOT print your answers). "
            "Use those to select the SINGLE BEST OPTION.\n\n"
            f"{prompt_text}\n\n"
            f"{options_only_line}\n\n"
            f"Follow-up Questions (context only):\n{followups_block.strip() or '(none)'}\n\n"
            "Output exactly one line:\nFINAL_ANSWER: <option or comma-separated options exactly from the listed Options>"
        )
        gen = self._generate_with(
            self.base_model, self.base_tok,
            self.system_prompt, user,
            max_new_tokens=64, do_sample=False, temperature=0.0,
            stopper=StoppingCriteriaList([stop_on_option_final_answer(self.base_tok)])
        )
        m = re.search(r'^\s*FINAL_ANSWER:\s*(.+?)\s*$', gen, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()

        user2 = (
            f"{prompt_text}\n\n{options_only_line}\n\n"
            "Output exactly one line:\nFINAL_ANSWER: <option or comma-separated options exactly from the listed Options>"
        )
        gen2 = self._generate_with(
            self.base_model, self.base_tok,
            "", user2, 64, False, 0.0
        )
        m2 = re.search(r'^\s*FINAL_ANSWER:\s*(.+?)\s*$', gen2, flags=re.IGNORECASE | re.MULTILINE)
        return m2.group(1).strip() if m2 else ""

    def decide_yesno(self, story: str, country: str, followups_block: str) -> str:
        user = (
            "First, read the follow-up questions and ANSWER THEM YOURSELF (internally; do NOT print your answers). "
            "Use those to REASON and select yes/no.\n\n"
            f"Scenario in {country}:\n{story}\n\n"
            f"Follow-up Questions (context only):\n{followups_block.strip() or '(none)'}\n\n"
            "Output exactly one line:\nFINAL_ANSWER: <YES|NO>"
        )
        gen = self._generate_with(
            self.base_model, self.base_tok,
            self.system_prompt, user,
            max_new_tokens=32, do_sample=False, temperature=0.0,
            stopper=StoppingCriteriaList([stop_on_yesno_final_answer(self.base_tok)])
        )
        m = re.search(r'FINAL_ANSWER:\s*(YES|NO)\b', gen.upper())
        if m:
            return m.group(1).upper()

        user2 = (
            f"Scenario in {country}:\n{story}\n\n"
            "Output exactly one line:\nFINAL_ANSWER: <YES|NO>"
        )
        gen2 = self._generate_with(
            self.base_model, self.base_tok,
            "", user2, 32, False, 0.0
        )
        m2 = re.search(r'FINAL_ANSWER:\s*(YES|NO)\b', gen2.upper())
        return (m2.group(1) if m2 else "").upper()

    @staticmethod
    def _strip_json_instruction(text: str) -> str:
        return re.sub(r'^\s*Answer\s+in\s+the\s+json\s+format.*$', "", text,
                      flags=re.IGNORECASE | re.MULTILINE).strip()

    @staticmethod
    def _extract_options_line(prompt_text: str) -> Tuple[str, List[str]]:
        m = re.search(r'Options:\s*(.*)', prompt_text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return "", []
        tail = m.group(1)
        first_non_empty = ""
        for ln in tail.splitlines():
            ln = ln.strip()
            if ln:
                first_non_empty = ln
                break
        if not first_non_empty:
            return "", []
        raw_opts = [o.strip() for o in first_non_empty.split(",") if o.strip()]
        return f"Options: {first_non_empty}", [o.lower() for o in raw_opts]


    def run_culturalbench(self,
                          culbench_csv: str,
                          out_csv: str = "culbench/culbench_adapter_followups_then_decide_reddit.csv"):
        raw = pd.read_csv(culbench_csv)
        # raw=raw[:10]
        raw.columns = [c.strip() for c in raw.columns]

        # Group rows into MCQs
        grouped = []
        for qid, g in raw.groupby("question_idx"):
            g = g.sort_values("data_idx")
            question = g.iloc[0]["prompt_question"]
            country = g.iloc[0].get("country", "")
            opts, correct_letter = [], ""
            for i, (_, row) in enumerate(g.iterrows()):
                letter = chr(65 + i)
                opts.append(f"{letter}) {row['prompt_option']}")
                if str(row["answer"]).upper() == "TRUE":
                    correct_letter = letter
            grouped.append({
                "question_idx": qid,
                "country": country,
                "prompt_question": question,
                "options": "\n".join(opts),
                "correct_option": correct_letter,
                "original_data_indices": g["data_idx"].tolist(),
            })
        qdf = pd.DataFrame(grouped)

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        self.adapter_model.set_adapter("culbench")
        followups_list = []
        for _, r in tqdm(qdf.iterrows(), total=len(qdf), desc="CulBench: follow-ups"):
            question_block = f"{r['prompt_question']}\n\nOptions:\n{r['options']}"
            prompt = self.build_followup_prompt(question_block)
            followups = self._generate_with(self.adapter_model, self.adapter_tok,
                                            self.system_prompt, prompt, 128, False, 0.0)
            followups_list.append(followups)

        rows = []
        for (idx, r), followups in tqdm(zip(qdf.iterrows(), followups_list), total=len(qdf), desc="CulBench: decide"):
            choice = self.decide_mcq_letter(r['prompt_question'], r['options'], followups)
            rows.append({
                **r,
                "follow_up_questions": followups,
                "model_choice": choice,
                "is_correct": (choice == r["correct_option"]) if r["correct_option"] else None
            })

        out = pd.DataFrame(rows)
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logging.info(f"Saved CulturalBench → {out_csv}")
        return out

    def run_commonsense(self,
                        in_csv: str,
                        out_csv: str = "results_cultural_common_sense/commonsense_adapter_followups_then_decide_reddit.csv",
                        language_col: str = "language",
                        country_col: str = "country",
                        prompt_col: str = "prompt",
                        answer_col: str = "answer",
                        filter_english: bool = False):
        df = pd.read_csv(in_csv)
        df.columns = [c.strip() for c in df.columns]
        if filter_english and language_col in df.columns:
            df = df[df[language_col].astype(str).str.lower() == "en"].copy()

        # Phase A: follow-ups (adapter model)
        self.adapter_model.set_adapter("commonsense")
        followups_list, prompts, options_lines, langs, countries, golds = [], [], [], [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Commonsense: follow-ups"):
            prompt_text = self._strip_json_instruction(str(row[prompt_col]))
            options_line, _ = self._extract_options_line(prompt_text)
            question_block = f"{prompt_text}"
            prompt = self.build_followup_prompt(question_block)
            followups = self._generate_with(self.adapter_model, self.adapter_tok,
                                            self.system_prompt, prompt, 128, False, 0.0)

            followups_list.append(followups)
            prompts.append(prompt_text)
            options_lines.append(options_line)
            langs.append(row.get(language_col, ""))
            countries.append(row.get(country_col, ""))
            golds.append(str(row.get(answer_col, "")))

        rows = []
        for i in tqdm(range(len(prompts)), desc="Commonsense: decide"):
            final_option = self.decide_option_text(prompts[i], options_lines[i], followups_list[i])
            rows.append({
                language_col: langs[i],
                country_col:  countries[i],
                prompt_col:   prompts[i],
                "gold_answer": golds[i],
                "model_answer": final_option,
                "follow_up_questions": followups_list[i],
            })

        out = pd.DataFrame(rows)
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logging.info(f"Saved Cultural Commonsense → {out_csv}")
        return out

    def run_normad(self,
                   out_csv: str = "normad/normad_adapter_followups_then_decide_reddit.csv",
                   story_col: str = "Story",
                   country_col: str = "Country",
                   source_csv: Optional[str] = None,
                   drop_neutral: bool = True):
        if source_csv:
            df = pd.read_csv(source_csv)
        else:
            normad_csv = hf_hub_download(
                repo_id="akhilayerukola/NormAd",
                filename="normad_etiquette_final_data.csv",
                repo_type="dataset"
            )
            df = pd.read_csv(normad_csv)
        df.columns = [c.strip() for c in df.columns]
        if drop_neutral and "Gold Label" in df.columns:
            df = df[df["Gold Label"].astype(str).str.lower() != "neutral"].copy()

        # Phase A: follow-ups (adapter model)
        self.adapter_model.set_adapter("normad")
        followups_list, countries, stories = [], [], []
        for _, r in tqdm(df.iterrows(), total=len(df), desc="NormAd: follow-ups"):
            country = str(r[country_col]); story = str(r[story_col])
            question_block = f"Country: {country}\nScenario: {story}"
            prompt = self.build_followup_prompt(question_block)
            followups = self._generate_with(self.adapter_model, self.adapter_tok,
                                            self.system_prompt, prompt, 128, False, 0.0)
            followups_list.append(followups); countries.append(country); stories.append(story)

        rows = []
        for i in tqdm(range(len(stories)), desc="NormAd: decide"):
            yn = self.decide_yesno(stories[i], countries[i], followups_list[i])
            rows.append({
                "Country": countries[i],
                "Story": stories[i],
                "model_answer": yn,
                "follow_up_questions": followups_list[i],
            })

        out = pd.DataFrame(rows)
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logging.info(f"Saved NormAd → {out_csv}")
        return out


if __name__ == "__main__":
    runner = AdapterRunner(
        base_model=BASE_MODEL,
        adapters_by_dataset=ADAPTERS_BY_DATASET,
        system_prompt=SYSTEM_PROMPT,
        use_4bit=USE_4BIT,
        dtype=DTYPE,
        device_map=DEVICE_MAP
    )

    runner.run_culturalbench(
        culbench_csv="hf://datasets/kellycyy/CulturalBench/CulturalBench-Hard.csv",
        out_csv="culbench/.csv"
    )

    runner.run_commonsense(
        in_csv="inputs_lang_country_en_culturalcommon.csv",
        out_csv="cultural_common_sense/.csv",
        language_col="language",
        country_col="country",
        prompt_col="prompt",
        answer_col="answer",
        filter_english=False
    )

    runner.run_normad(
        out_csv="normad/.csv",
        story_col="Story",
        country_col="Country",
        source_csv=None,        # or path to local NormAd CSV
        drop_neutral=True
    )
