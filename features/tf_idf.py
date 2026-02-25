import re
from collections import Counter
from tqdm import tqdm

import torch


class TF_IDF:
    def __init__(self, data, capacity=10000):
        """
        Counting most ``capacity`` frequent words in train data and saving it
        :param capacity:
        """
        words_string = "".join(data).lower()
        words_list = re.sub(r'[^a-zA-Z ]', '', words_string).split()
        print(len(words_list))  # 5 миллионов слов!
        counter = Counter(words_list)
        counter_list = list(counter.items())
        counter_list.sort(key=lambda x: x[1], reverse=True)
        counter_list = counter_list[:capacity]
        words, _ = zip(*counter_list)
        words = list(words)
        self.words = words
        self.capacity = capacity

        self.dfs = torch.zeros(capacity)

        self.word_to_index = {w: i for i, w in enumerate(words)}
        N = len(data)
        for comment in tqdm(data):
            comment_words = re.sub(r'[^a-zA-Z ]', '', comment.lower()).split()
            for word in set(comment_words):
                index = self.word_to_index.get(word)
                if index is not None:
                    self.dfs[index] += 1


        self.idfs = torch.log((N+1) / (self.dfs+1)) + 1


    def transform_one(self, text):
        ans = torch.zeros(self.capacity)
        transformed_text = re.sub(r'[^a-zA-Z ]', '', text.lower()).split()
        N = len(transformed_text)
        if N == 0:
            return ans

        words = Counter(transformed_text)

        for word, count in words.items():
            index = self.word_to_index.get(word)
            if index is not None:
                ans[index] = count / N * self.idfs[index]

        return ans

    def transform_batch(self, texts):
        return torch.stack([self.transform_one(t) for t in texts])