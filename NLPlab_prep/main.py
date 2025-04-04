import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from training import get_metrics_report
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 100

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
#DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

for DATASET in ["MR", "Semeval2017A"]:
    # load the raw data
    if DATASET == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif DATASET == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # convert data labels from strings to integers
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    n_classes = len(label_encoder.classes_)

    # Debug: print first 10 labels and their encoding
    print("\n[INFO] First 10 labels (encoded):")
    for i in range(10):
        print(f"{label_encoder.inverse_transform([y_train[i]])[0]} -> {y_train[i]}")

    # Define our PyTorch-based Dataset
    train_set = SentenceDataset(X_train, y_train, word2idx)
    test_set = SentenceDataset(X_test, y_test, word2idx)

    # EX7 - Define our PyTorch-based DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    #############################################################################
    # Model Definition (Model, Loss Function, Optimizer)
    #############################################################################
    if n_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
        output_size=1
    else:
        criterion = torch.nn.CrossEntropyLoss()
        output_size=n_classes

    model = BaselineDNN(output_size=output_size,
                        embeddings=embeddings,
                        trainable_emb=EMB_TRAINABLE)

    # move the mode weight to cpu or gpu
    model.to(DEVICE)
    print(model)

    # We optimize ONLY those parameters that are trainable (p.requires_grad==True)
    # 1. Επιλογή loss function
    #
    # 2. Επιλογή trainable παραμέτρων
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # 3. Επιλογή optimizer (π.χ. Adam)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    #############################################################################
    # Training Pipeline
    #############################################################################
    train_losses = []
    test_losses = []

    for epoch in range(1, EPOCHS + 1):
        # train the model for one epoch
        train_dataset(epoch, train_loader, model, criterion, optimizer)

        # evaluate the performance of the model, on both data sets
        train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                                model,
                                                                criterion)

        test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                            model,
                                                            criterion)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(get_metrics_report(y_test_gold, y_test_pred))

    print("\n\nFinal Evaluation on Test Set:")
    print(get_metrics_report(y_test_gold, y_test_pred))


    epochs = list(range(1, EPOCHS + 1))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()
