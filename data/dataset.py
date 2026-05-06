from pathlib import Path
import joblib
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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = torch.tensor(self.labels[idx])
        return text, y


class RNNDataset(Dataset):   # TODO: кажется почти ничем не отличается от CommentDataset
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.texts.iloc[idx]
        y = torch.tensor(self.labels[idx])
        return x, y


def split_data(X, y, test_size = 0.2, val_size = 0.0, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size + val_size,
        random_state=42,
        shuffle=shuffle,
        stratify=y
    )
    if val_size > 1e-9:
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test,
            test_size=val_size / (test_size + val_size),
            random_state=42,
            shuffle=shuffle,
            stratify=y_test
        )
        return X_train, X_test, X_val, y_train, y_test, y_val
    return X_train, X_test, y_train, y_test


def get_data_loaders(preprocessor, batch_size=64, capacity=10000, test_size=0.2, val_size=0.2):
    match preprocessor:
        case "bow":
            data = pd.read_csv(BASE_DIR / "processed" / "train.csv")
            X_train, X_test, y_train, y_test = split_data(data['comment_text'], data['negative'], test_size=test_size)

            bow = BagOfWords(data=X_train.tolist(), capacity=capacity)

            X_train = bow.transform_batch(X_train)
            X_test = bow.transform_batch(X_test)
            train_loader = DataLoader(CommentDataset(X_train, y_train))
            test_loader = DataLoader(CommentDataset(X_test, y_test))

            return train_loader, test_loader
        case "tfidf":
            data = pd.read_csv(BASE_DIR / "processed" / "train.csv")
            X_train, X_test, X_val, y_train, y_test, y_val  = split_data(data['comment_text'], data['negative'],
                                                                         test_size=test_size, val_size=val_size)
            PATH = Path('tfidf.pt')

            if PATH.exists():
                tfidf = TF_IDF.load(PATH)
            else:
                tfidf = TF_IDF(X_train, capacity=capacity)
                tfidf.save(PATH)

            X_train = tfidf.transform_batch(X_train)
            X_test = tfidf.transform_batch(X_test)
            X_val = tfidf.transform_batch(X_val)

            train_loader = DataLoader(CommentDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(CommentDataset(X_val, y_val), batch_size=batch_size)
            test_loader = DataLoader(CommentDataset(X_test, y_test), batch_size=batch_size)

            return train_loader, val_loader, test_loader

        case "vocab_left_pad" | "vocab_right_pad":   # TODO: починить порядок
            train_df = joblib.load(BASE_DIR / preprocessor / "train.pkl")
            test_df = joblib.load(BASE_DIR / preprocessor / "test.pkl")
            val_df = joblib.load(BASE_DIR / preprocessor / "val.pkl")

            train_dataset = RNNDataset(train_df['tokens'], train_df['toxic'])
            test_dataset = RNNDataset(test_df['tokens'], test_df['toxic'])
            val_dataset = RNNDataset(val_df['tokens'], val_df['toxic'])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            return train_loader, val_loader, test_loader
        case _:
            raise NotImplementedError(f"Data loader for specified data preprossecor <{preprocessor}> is not implemented")

def get_corpus():
    data = pd.read_csv(BASE_DIR / "processed" / "train.csv")
    return split_data(data['comment_text'], data['negative'])
