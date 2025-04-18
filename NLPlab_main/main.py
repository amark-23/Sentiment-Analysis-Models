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
from models import *
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from training import torch_train_val_split
from early_stopper import EarlyStopper
from attention import *

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

    # Split training set into training + validation
    train_loader, val_loader = torch_train_val_split(
        train_set,
        batch_train=BATCH_SIZE,
        batch_eval=BATCH_SIZE,
        val_size=0.2
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --- Loss function ---
    if n_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
        output_size = 1
    else:
        criterion = torch.nn.CrossEntropyLoss()
        output_size = n_classes

    # --- Model ---
    
    """ #BaselineDNN
    model = BaselineDNN(
    output_size=output_size,
    embeddings=embeddings,
    trainable_emb=EMB_TRAINABLE
    ).to(DEVICE)    
    """
    
    """ #LSTM
    model = LSTM(
        output_size=output_size,
        embeddings=embeddings,
        trainable_emb=EMB_TRAINABLE,
        bidirectional=True
    ).to(DEVICE)
    """

    """ #SimpleAttention
    model = SimpleSelfAttentionModel(
    output_size=output_size,
    embeddings=embeddings
    ).to(DEVICE)
    """

    """ #MultiHeadAttention
    model = MultiHeadAttentionModel(
    output_size=output_size,
    embeddings=embeddings,
    max_length=60,     
    n_head=5           
    ).to(DEVICE)
    """
    
    # --- Model ---
    model = TransformerEncoderModel(
    output_size=output_size,
    embeddings=embeddings,
    max_length=60,
    n_head=3,        # 4, 5
    n_layer=3        # 2, 6 ανάλογα
    ).to(DEVICE)



    # --- Early Stopper (after model is defined!) ---
    save_path = f"best_model_{DATASET}.pt"
    early_stopper = EarlyStopper(model, save_path, patience=5, min_delta=0.001)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    # Training Pipeline
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_dataset(epoch, train_loader, model, criterion, optimizer)
        val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader, model, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(get_metrics_report(y_val_gold, y_val_pred))

        if early_stopper.early_stop(val_loss):
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
            break

    # Τελική αξιολόγηση στο test set
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, criterion)
    print("\n\nFinal Evaluation on Test Set:")
    print(get_metrics_report(y_test_gold, y_test_pred))

    # Plot
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss ({DATASET})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
