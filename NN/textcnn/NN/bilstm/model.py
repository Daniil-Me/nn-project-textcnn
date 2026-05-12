import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size,
                 num_layers, num_classes, dropout, embedding_matrix):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden))
