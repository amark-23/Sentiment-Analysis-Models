# Speech and Language Processing Labs - Sentiment Classification with Deep Learning & Transformers

This project implements and compares various sentiment classification models on the MR and SemEval2017A datasets. Techniques range from simple baselines to advanced pre-trained transformer fine-tuning.

## Models Implemented
- Baseline DNN
- LSTM (uni/bi-directional)
- Self-Attention & Multi-Head Attention
- Transformer Encoder
- Pre-trained models via Hugging Face (`siebert/`, `cardiffnlp/`, etc.)
- Fine-tuned Transformer models (Colab-compatible)

## ⚙️ Features
- Embedding loading (e.g. GloVe)
- Attention mechanisms from scratch
- Evaluation with accuracy, recall, F1-score
- Hugging Face Transformers pipeline & fine-tuning
- Modular code structure (models, training, utils)

## Datasets
- MR (Movie Reviews, binary)
- SemEval2017 Task 4 (Twitter sentiment, 3-class)

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Run experiments or fine-tune models using:
```bash
python main.py       # for classic models
python transfer_pretrained.py  # for zero-shot
python finetuned_pretrained.py # pretrained models finetuning 
```

Fine-tuning can also be done in Colab (`finetune_pretrained.py`).

# Repo structure

```

├── NLPlab          # Helper code for NLP lab
├── LAB-SUPPORT.md  # Instructions on how to use Github's issue tracking to submit questions to the TAs
├── LICENSE         # MIT License
└── README.md       # This file

```

