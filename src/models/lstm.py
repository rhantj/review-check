import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        e = self.emb(x)
        _, (h, _) = self.lstm(e)
        return self.fc(h[-1])
