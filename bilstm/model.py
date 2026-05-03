import torch
import torch.nn as nn
from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT, NUM_CLASSES, EMBEDDING_DIM


class BiLSTM(nn.Module):

    def __init__(self, embedding_matrix):
        super(BiLSTM, self).__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.FloatTensor(embedding_matrix),
            requires_grad=False
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_forward = hidden[-2, :, :] if NUM_LAYERS > 1 else hidden[0, :, :]
        hidden_backward = hidden[-1, :, :]
        combined = torch.cat([hidden_forward, hidden_backward], dim=1)
        dropped = self.dropout(combined)
        output = self.fc(dropped)
        return output