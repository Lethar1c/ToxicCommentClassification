import re
from typing import List

import spacy
import torch
from torch import Tensor

spacy.prefer_gpu()

class Vocabulary:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.__last_index = 1

    def build(self, corpus: List[str]):
        corpus = [re.sub(r'[^a-zA-Z0-9\'\- ]', '', x) for x in corpus]
        for text in corpus:
            doc = self.nlp(text)
            for token in doc:
                if not token.text.lower() in self.word_to_idx.keys():
                    word = token.text.lower()
                    self.__last_index += 1
                    self.word_to_idx[word] = self.__last_index

    def encode_one(self, text: str, max_len=150, padding=True) -> Tensor:
        doc = self.nlp(text)
        ans = []
        for token in doc:
            ans.append(self.word_to_idx.get(token.text.lower(), 1))
            if len(ans) >= max_len:
                break
        if padding:
            return torch.tensor([0] * (max_len - len(ans)) + ans)
        return torch.tensor(ans)

    def encode(self, texts: list[str], max_len=150) -> Tensor:
        return torch.stack([self.encode_one(x, max_len) for x in texts])

    def __len__(self):
        return len(self.word_to_idx)