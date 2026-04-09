from typing import Iterable

from torch import nn, Tensor

from data import dataset
from features.tokenizer import Vocabulary


class RNNModel(nn.Module):
    def __init__(self, vocabulary: Vocabulary, embedding_dim=128, rnn_hidden_size=128):
        super().__init__()
        # self.vocabulary = Vocabulary()
        # self.vocabulary.build(dataset.get_corpus()[0])
        self.vocabulary = vocabulary
        self.__vocab_size = len(self.vocabulary)
        self.embedding = nn.Embedding(self.__vocab_size, embedding_dim, 0)
        self.rnn = nn.RNN(embedding_dim, rnn_hidden_size, batch_first=True)
        self.classifier = nn.Linear(rnn_hidden_size, 2)

    def forward(self, x: Tensor):
        # x = list(map(lambda t: self.vocabulary.encode(t), x))
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        return self.classifier(hidden[-1])

