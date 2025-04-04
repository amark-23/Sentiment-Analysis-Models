import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        embeddings = np.array(embeddings)
        num_embeddings, emb_dim = embeddings.shape
        self.embedding = nn.Embedding(num_embeddings, emb_dim)

        # 2 - initialize the weights of our Embedding layer from pretrained embeddings
        self.embedding.weight.data.copy_(torch.tensor(embeddings))

        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding.weight.requires_grad = trainable_emb
        
        # 4 - define a non-linear transformation of the representations
        self.hidden = nn.Linear(2 * emb_dim, 64)  # using mean, min, max → emb_dim * 3
        self.activation = nn.ReLU()

        # 5 - define the final Linear layer which maps the representations to the classes
        self.output = nn.Linear(64, output_size)

    def forward(self, x, lengths):
        """
        Forward pass using [mean || max] pooling over word embeddings.

        Args:
            x (Tensor): tensor of shape (batch_size, max_len) containing word indices
            lengths (Tensor): actual lengths of each sentence (no padding)

        Returns:
            logits (Tensor): tensor of shape (batch_size, num_classes)
        """

        embeddings = self.embedding(x)  # (batch_size, max_len, emb_dim)
        mask = (x != 0).unsqueeze(-1)   # (batch_size, max_len, 1)

        # --- Mean pooling ---
        summed = torch.sum(embeddings * mask, dim=1)  # sum of embeddings (excluding pads)
        lengths = lengths.unsqueeze(1)  # (batch_size, 1)
        mean_pool = summed / lengths  # (batch_size, emb_dim)

        # --- Max pooling ---
        masked_embeddings = embeddings.masked_fill(mask == 0, float('-inf'))
        max_pool = masked_embeddings.max(dim=1).values  # (batch_size, emb_dim)

        # --- Concatenate mean and max ---
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (batch_size, 2 * emb_dim)

        # --- Hidden layer + activation ---
        representations = self.activation(self.hidden(pooled))  # (batch_size, 64)

        # --- Final output projection ---
        logits = self.output(representations)  # (batch_size, num_classes)

        if self.output.out_features == 1:
            logits = logits.squeeze(1)

        return logits



class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        representations = ...

        logits = self.linear(representations)
        if self.output.out_features == 1:
            logits = logits.squeeze(1)
        return logits
