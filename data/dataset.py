from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

from features.bag_of_words import BagOfWords
from features.tf_idf import TF_IDF

BASE_DIR = Path(__file__).resolve().parent

class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels.reset_index(drop=True)
        # self.vectorizer = vectorizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # x = self.vectorizer.transform_one(text)
        y = torch.tensor(self.labels[idx])
        return text, y


def get_bow_data_loaders(batch_size=64):
    data = pd.read_csv(BASE_DIR / "processed" / "train.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data['comment_text'],
        data['negative'],
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    bow = BagOfWords(data=X_train.tolist(), capacity=10000)

    X_train = bow.transform_batch(X_train)
    X_test = bow.transform_batch(X_test)

    train_dataset = CommentDataset(X_train, y_train)
    test_dataset = CommentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def get_tfidf_data_loaders(batch_size=64, capacity=10000):
    data = pd.read_csv(BASE_DIR / "processed" / "train.csv")


    X_train, X_test, y_train, y_test = train_test_split(
        data['comment_text'],
        data['negative'],
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    tfidf = TF_IDF(X_train, capacity=capacity)

    X_train = tfidf.transform_batch(X_train)
    X_test = tfidf.transform_batch(X_test)

    train_dataset = CommentDataset(X_train, y_train)
    test_dataset = CommentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
