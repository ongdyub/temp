import torch
import torch.nn as nn

class ChordLSTM(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(output)
        return logits


class ChordBiLSTM(nn.Module):
    def __init__(self, vocab_size=140, embedding_dim=2048, hidden_dim=512, num_layers=5):
        super(ChordBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(output)
        return logits
# Initialize the model
# model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
