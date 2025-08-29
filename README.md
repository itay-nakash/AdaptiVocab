# AdaptiVocab Codebase

> **Paper:** *AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation* — Itay Nakash, Nitay Calderon, Eyal Ben David, Elad Hoffer, Roi Reichart (arXiv:[2503.19693](https://arxiv.org/abs/2503.19693)), Accepted to COLM 2025.

---
## Installation

### 1. Create a Conda environment
We recommend using **Python 3.10**:

```bash
conda create -n adaptivocab python=3.10 -y
conda activate adaptivocab

```
### 2.Install PyTorch

### 3. Install required packages
```bash
pip install -r requirements.txt
```

## Overview

This repo implements **AdaptiVocab**, a framework to adapt the tokenizer of large language models for domain efficiency.  
Pipeline:

1. **Build PatchTokenizer** — remove rare tokens, add domain-specific n-grams, create initial embbeding.
2. **Patch & fine-tune the base model** — adjust embeddings and train with the new tokenizer.  
3. **Evaluate outputs** — LLM-as-judge scoring, perplexity, and human evaluation.

---

## Main Scripts

- **`create_patch_tokenizer.py`**  
  Builds a PatchTokenizer from a target corpus, saves artifacts (`config.pkl`, `removed_tokens.pkl`, `added_ngrams.pkl`, `patch_tokenizer.pkl`).  

- **`ft_model_new.py`**  
  Loads model + patched tokenizer, remaps embeddings, and fine-tunes.

- **`evaluate_model_outputs_new.py`**  
  Processes saved JSON outputs, computes perplexity, and runs LLM-as-judge scoring (Mistral or Gemini). Exports aggregated CSVs.


---
