import numpy as np
import os
os.environ["WANDB_DISABLED"] = "true"
import evaluate
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from utils.load_datasets import load_MR, load_Semeval2017A

# === SELECT DATASET ===
DATASET = 'MR'  # or 'Semeval2017A'

# === Define models per dataset ===
MODEL_LIST = {
    "MR": [
        'siebert/sentiment-roberta-large-english',
        "bert-base-multilingual-cased"
        'distilbert-base-uncased-finetuned-sst-2-english',
        'AnkitAI/reviews-roberta-base-sentiment-analysis'
    ],
    "Semeval2017A": [
        'cardiffnlp/twitter-roberta-base-sentiment',
        'finiteautomata/bertweet-base-sentiment-analysis',
        'j-hartmann/sentiment-roberta-large-english-3-classes'
    ]
}

# === Evaluation metric ===
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def prepare_dataset(X, y):
    return Dataset.from_dict({'text': X, 'label': y})

# === Load dataset ===
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# === Encode labels ===
le = LabelEncoder()
le.fit(list(set(y_train)))
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)
n_classes = len(le.classes_)

# === Prepare HuggingFace datasets ===
train_set = prepare_dataset(X_train, y_train_enc)
test_set = prepare_dataset(X_test, y_test_enc)

# === Loop over models ===
for PRETRAINED_MODEL in MODEL_LIST[DATASET]:
    print(f"\n Fine-tuning: {PRETRAINED_MODEL}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, num_labels=n_classes)

    # Tokenize
    tokenized_train = train_set.map(tokenize_function)
    tokenized_test = test_set.map(tokenize_function)
    
    
    # Small subset for quick training (just for testing in Colab)
    small_train = tokenized_train.shuffle(seed=42).select(range(40))
    small_eval = tokenized_test.shuffle(seed=42).select(range(40))
    

    # Training setup
    args = TrainingArguments(
        output_dir=f"{PRETRAINED_MODEL.replace('/', '_')}_output",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_steps=5,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train,
        eval_dataset=small_eval,
        compute_metrics=compute_metrics,
    )

    # Fine-tune
    trainer.train()

    # Evaluate
    print(" Evaluation:")
    trainer.evaluate()
