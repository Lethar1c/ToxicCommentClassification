from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size=256, out_size=1, embedding_dim=128, num_layers=1, p=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self._vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, (hidden, c) = self.lstm(emb)
        drop = self.dropout(hidden[-1])
        logits = self.classifier(drop)
        return logits


