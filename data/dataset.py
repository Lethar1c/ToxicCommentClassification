from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

from features.bag_of_words import BagOfWords

BASE_DIR = Path(__file__).resolve().parent

class CommentDataset(Dataset):
    def __init__(self, texts, labels, bow):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.bow = bow

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        x = self.bow.transform_one(text)
        y = torch.tensor(self.labels[idx])
        return x, y

def get_data_loaders(batch_size=64):
    data = pd.read_csv(BASE_DIR / "processed" / "train.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data['comment_text'],
        data['negative'],
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 🔥 фитим словарь только на train
    bow = BagOfWords(data=X_train.tolist(), capacity=10000)

    train_dataset = CommentDataset(X_train, y_train, bow)
    test_dataset = CommentDataset(X_test, y_test, bow)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

