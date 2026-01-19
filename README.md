# Multilingual Voice AI Intent Classification

## Project Overview

This project presents an **intent classification solution** for a multilingual Voice AI system, designed to handle English, Kinyarwanda (`rw`), and code-switched utterances. The system leverages **XLM-RoBERTa (`xlm-roberta-base`)** to classify user intents from ASR-transcribed speech, including informal phrasing, borrowed words, and low-resource language utterances.

The main goals of this project are to:

- Demonstrate handling of **low-resource languages** and code-switching.
- Provide a robust **PyTorch implementation** for training, evaluation, and inference.
- Offer a **language-aware evaluation strategy** to ensure fairness and performance transparency.
- Outline **production-ready considerations**, including confidence-based fallback, monitoring, and MLOps best practices.

---

## Repository Contents

| File / Directory | Description |
|-----------------|------------|
| `Intent_Classification_Notebook.ipynb` | Complete step-by-step implementation, including data preprocessing, model training, evaluation, and visualization. |
| `Summary_Report.pdf` | Concise written summary (≤4 pages) covering approach, assumptions, trade-offs, evaluation, and ethics. |
| `inference_pipeline/` | Example code for inference, showing how to process a new utterance and predict intent in real-time. |
| `data/` | Sample dataset CSVs used for training and validation. |

---

## Inference Pipeline

The project includes a **ready-to-use inference pipeline**, demonstrating how to:

1. Load the trained XLM-R intent classification model.
2. Tokenize and preprocess new utterances using `XLMRobertaTokenizer`.
3. Predict intents with softmax probabilities.
4. Apply **confidence-based fallback** for low-confidence predictions.
5. Integrate predictions into a Voice AI system.

This pipeline is structured for easy extension and integration into **real-time Voice AI applications**.

---

## How to Use

1. Open the notebook `intent_classification_notebook.ipynb` to see **full code and methodology**.
2. View `docs/summary_report.pdf` for a **concise explanation** of design choices, assumptions, and results.
3. Use the `app/` directory to test the inference api
    - `python3 -m venv venv`
    - `source venv/bin/activate`
    -  `pip install -r requiremnets.txt`
    - `uvicorn inference_pipeline.app:app --reload --host 0.0.0.0 --port 8000`
---

## Notes

- All experiments are reproducible with the provided notebook and dataset.
- The model and tokenizer can be saved and reloaded for **offline evaluation or production deployment**.
