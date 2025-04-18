from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report

# Επιλογή dataset
DATASET = 'Semeval2017A'  # 'MR' or 'Semeval2017A'

# Προκαθορισμένα μοντέλα ανά dataset
MODELS = {
    'MR': [
        'siebert/sentiment-roberta-large-english',
        'distilbert-base-uncased-finetuned-sst-2-english',
        'AnkitAI/reviews-roberta-base-sentiment-analysis',
    ],
    'Semeval2017A': [
        'cardiffnlp/twitter-roberta-base-sentiment',
        'finiteautomata/bertweet-base-sentiment-analysis',
        'j-hartmann/sentiment-roberta-large-english-3-classes',
    ]
}

# Αντιστοίχιση ετικετών ανά μοντέλο
LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {'POSITIVE': 'positive', 'NEGATIVE': 'negative'},
    'distilbert-base-uncased-finetuned-sst-2-english': {'POSITIVE': 'positive', 'NEGATIVE': 'negative'},
    'AnkitAI/reviews-roberta-base-sentiment-analysis': {'POSITIVE': 'positive', 'NEGATIVE': 'negative'},
    
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'finiteautomata/bertweet-base-sentiment-analysis': {
        'NEG': 'negative',
        'NEU': 'neutral',
        'POS': 'positive',
    },
    'j-hartmann/sentiment-roberta-large-english-3-classes': {
        'NEGATIVE': 'negative',
        'NEUTRAL': 'neutral',
        'POSITIVE': 'positive',
    }
}

if __name__ == '__main__':
    # Φόρτωση dataset
    if DATASET == 'MR':
        X_train, y_train, X_test, y_test = load_MR()
        label_set = ['positive', 'negative']
    elif DATASET == 'Semeval2017A':
        X_train, y_train, X_test, y_test = load_Semeval2017A()
        label_set = ['positive', 'negative', 'neutral']
    else:
        raise ValueError("Unsupported dataset")

    # Encode labels
    le = LabelEncoder()
    le.fit(label_set)
    y_test_enc = le.transform(y_test)

    for model_name in MODELS[DATASET]:
        print(f"\nEvaluating model: {model_name}")
        sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

        y_pred = []
        for text in tqdm(X_test):
            prediction = sentiment_pipeline(text[:512])[0]['label']
            prediction = prediction.upper().strip()  # Κάνουμε τη σύγκριση case-insensitive
            mapped = LABELS_MAPPING[model_name][prediction]
            y_pred.append(mapped)

        y_pred_enc = le.transform(y_pred)

        print(f"\nResults for model: {model_name}")
        print(get_metrics_report([y_test_enc], [y_pred_enc]))
