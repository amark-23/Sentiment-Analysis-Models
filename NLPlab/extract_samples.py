import os
import random
from collections import defaultdict
from utils.load_datasets import load_Semeval2017A, load_MR

# Create top-level Samples folder if it doesn't exist
os.makedirs("Samples", exist_ok=True)

for DATASET in ["MR", "Semeval2017A"]:
    print(f"\n Processing dataset: {DATASET}")

    # Load dataset & label set
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
        label_set = ['positive', 'negative', 'neutral']
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
        label_set = ['positive', 'negative']
    else:
        raise ValueError("Unknown dataset")

    # Combine train/test
    all_texts = X_train + X_test
    all_labels = y_train + y_test

    # Group by label
    examples = defaultdict(list)
    for text, label in zip(all_texts, all_labels):
        examples[label.lower()].append(text.strip())

    # Prepare output folders
    dataset_folder = os.path.join("Samples", "Semeval" if "Sem" in DATASET else "MR")
    os.makedirs(dataset_folder, exist_ok=True)

    # Collect (label, sentence) pairs
    labeled_pairs = []

    for label in label_set:
        sampled = random.sample(examples[label], min(20, len(examples[label])))
        for sentence in sampled:
            labeled_pairs.append((label, sentence.strip()))

    # Shuffle the pairs once
    random.shuffle(labeled_pairs)

    # Prepare lines for each file
    labeled_lines = [f"{label} --- {sentence}\n" for label, sentence in labeled_pairs]
    unlabeled_lines = [f"{sentence}\n" for _, sentence in labeled_pairs]

    # Filenames
    base_name = f"sampled_sentences_{DATASET}"
    labeled_path = os.path.join(dataset_folder, f"{base_name}.txt")
    nolabels_path = os.path.join(dataset_folder, f"{base_name}_nolabels.txt")

    # Save labeled
    with open(labeled_path, "w", encoding="utf-8") as f:
        f.writelines(labeled_lines)

    # Save unlabeled (same order)
    with open(nolabels_path, "w", encoding="utf-8") as f:
        f.writelines(unlabeled_lines)

    print(f"  Saved: {labeled_path}")
    print(f"  Saved: {nolabels_path}")
