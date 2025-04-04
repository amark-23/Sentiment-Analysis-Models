import os
from sklearn.metrics import accuracy_score

def evaluate_chatgpt(dataset_name):
    folder = "Samples/MR" if dataset_name == "MR" else "Samples/Semeval"

    true_file = os.path.join(folder, f"sampled_sentences_{dataset_name}.txt")
    gpt_file = os.path.join(folder, f"sampled_sentences_{dataset_name}_chatgpt.txt")

    with open(true_file, "r", encoding="utf-8") as f:
        true_lines = [line.strip() for line in f if '---' in line]

    true_labels = []
    sentences = []
    for line in true_lines:
        parts = line.split('---')
        true_labels.append(parts[0].strip().lower())
        sentences.append(parts[1].strip())

    with open(gpt_file, "r", encoding="utf-8") as f:
        gpt_lines = [line.strip() for line in f if '---' in line]

    pred_labels = [line.split('---')[0].strip().lower() for line in gpt_lines]

    if len(true_labels) != len(pred_labels):
        print(f"\n")
        print(f"Mismatch in number of lines for {dataset_name}:")
        print(f"True labels: {len(true_labels)}")
        print(f"GPT labels: {len(pred_labels)}")
        return

    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nAccuracy for {dataset_name}: {accuracy * 100:.2f}%")

    print("\nSample mismatches:")
    mismatch_count = 0
    for t, p, s in zip(true_labels, pred_labels, sentences):
        if t != p:
            print(f"TRUE: {t} | GPT: {p}")
            print(f"TEXT: {s}\n")
            mismatch_count += 1
            if mismatch_count >= 5:
                break

evaluate_chatgpt("MR")
evaluate_chatgpt("Semeval2017A")
