import re
from collections import Counter

import torch


class TF_IDF:
    def __init__(self, data, capacity=10000):
        """
        Counting most ``capacity`` frequent words in train data and saving it
        :param capacity:
        """
        # data = get_all_comments()
        words_string = "".join(data).lower()
        words_list = re.sub(r'[^a-zA-Z ]', '', words_string).split()
        print(len(words_list))  # 5 миллионов слов!
        counter = Counter(words_list)
        counter_list = list(counter.items())
        counter_list.sort(key=lambda x: x[1], reverse=True)
        counter_list = counter_list[:capacity]
        words, _ = zip(*counter_list)
        words = list(words)
        words.sort()
        self.words = words
        self.capacity = capacity

    def transform_one(self, text):
        ans = torch.zeros(self.capacity)

        words = re.sub(r'[^a-zA-Z ]', '', text.lower()).split()
        for i, w in enumerate(self.words):
            if w in words:
                ans[i] = 1

        return ans

    def transform_batch(self, texts):
        return torch.stack([self.transform_one(t) for t in texts])