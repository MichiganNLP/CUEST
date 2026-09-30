# The Curious Case of Curiosity across Cultures
### Evaluating Cross-Cultural Information-Seeking Questions in Humans and LLMs

**Angana Borah**, **Zhijing Jin**, and **Rada Mihalcea**

Accepted to **AACL 2026 Main Conference**.

[Project Website](https://lit.eecs.umich.edu/CUEST/) 

## Overview

When people seek to understand the world, they ask questions. The questions people choose to ask—and how they phrase them—can vary across cultural and social contexts. Most cross-cultural evaluations of large language models focus on their answers. This work instead studies the step before an answer: **what information models choose to seek**. We introduce **IQUEST** (*Information-seeking QUestion Evaluation across SocieTies*), a framework for comparing human and LLM curiosity-like question-asking behavior across cultures.

## Why Information-Seeking Matters

A model’s ability to ask appropriate questions can affect its usefulness, engagement, and sensitivity to cultural context. Models that fail to seek relevant context may silently rely on familiar assumptions, potentially flattening cultural differences. Culture-aware systems should therefore do more than memorize cultural facts. They should also learn **when and how to ask questions**.

<img width="479" height="429" alt="Screenshot 2026-09-30 at 7 36 26 PM" src="https://github.com/user-attachments/assets/45da394f-5587-4ca2-b5b8-0523ec99f1e0" />


## IQUEST Framework

IQUEST evaluates human and model-generated questions along three complementary dimensions:

1. **Linguistic alignment**: Measures differences in ambiguity, open-endedness, rhetorical devices, and cohesion.

2. **Topic-preference alignment**: Compares the topics that humans and language models prioritize across countries.

3. **Social-science grounding**: Relates information-seeking patterns to established cultural values, contextual communication, and education systems.

Our evaluation covers:

- 18 countries
- 16 shared topics
- Human-authored and LLM-generated questions
- Open- and closed-source language models
- Three downstream cultural-adaptability benchmarks

## Main Findings

- **LLMs flatten cross-cultural diversity.** Human question-asking patterns vary more across countries than model-generated questions.
- **Models align more closely with Western question patterns.** Larger human–model gaps appear in several Eastern and Latin American contexts.
- **Question style and topic preference capture different behaviors.** A model can resemble humans linguistically while prioritizing different subjects.
- **Cultural prompting helps, but is insufficient.** Country personas reduce some gaps without fully reproducing human information-seeking patterns.
- **Fine-tuning improves alignment.** Adapter-based fine-tuning reduces the overall human–model linguistic alignment gap by approximately **43%**.
- **Better questions improve cultural reasoning.** Information-seeking variants improve performance across NormAd, CulturalBench, and Cultural Commonsense.

## Repository Structure

```text
CUEST/
├── adapter_ft.py
├── adapter_inference_test.py
├── cbench_prompt_ask.py
├── downstream_prompt_ft.py
├── gpt4o_statements_obj2.py
├── linguistic_analysis.py
├── data/
└── docs/
    ├── index.html
    ├── styles.css
    ├── script.js
    └── assets/
        └── paper.pdf
```

### Main files

- `linguistic_analysis.py` — linguistic analysis of human and model-generated questions
- `adapter_ft.py` — adapter-based fine-tuning
- `adapter_inference_test.py` — inference and evaluation using trained adapters
- `cbench_prompt_ask.py` — information-seeking evaluation on CulturalBench
- `downstream_prompt_ft.py` — downstream cultural-adaptability experiments
- `gpt4o_statements_obj2.py` — data generation for the conversational information-seeking objective
- `data/` — released data and experiment artifacts
- `docs/` — static project website

## Project Website

The project website is available at:

**https://lit.eecs.umich.edu/CUEST/**

## Citation

If you use this work, please cite: [TO BE UPDATED ONCE AACL PROCEEDINGS ARE OUT]

```bibtex
@misc{borah2026curious,
  title  = {The Curious Case of Curiosity across Cultures:
            Evaluating Cross-Cultural Information-Seeking Questions
            in Humans and LLMs},
  author = {Borah, Angana and Jin, Zhijing and Mihalcea, Rada},
  year   = {2026},
  note   = {Accepted to AACL 2026 Main Conference}
}
```

## Contact

For questions, please contact Angana Borah: `anganab@umich.edu`.
