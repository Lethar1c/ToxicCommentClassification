from pathlib import Path
import joblib

import torch.optim
from torch import nn
from torchmetrics.classification import BinaryF1Score

from data.dataset import get_corpus, get_data_loaders
from models.MLP.model import MLPModel
from models.LogisticRegression.model import LogisticRegressionModel
from models.RNN.model import RNNModel
from models.LSTM.model import LSTMModel
from features.tokenizer import Vocabulary
from metrics.metrics import get_regression_metrics, find_best_threshold
from training.trainer import Trainer

EPOCHES = 5
device = "cuda" if torch.cuda.is_available() else 'cpu'
VOCABULARY_PATH = Path("saves") / "vocabulary.pkl"

def get_vocabulary(X_train):
    try:
        print("loading vocabulary")
        vocabulary = joblib.load(VOCABULARY_PATH)
    except Exception:
        print("initializing vocabulary")
        vocabulary = Vocabulary()
        vocabulary.build(X_train)
        print("vocabulary initialized. saving...")
        joblib.dump(vocabulary, VOCABULARY_PATH)
    return vocabulary

def train_model(model_name, preprocessor, test_size = 0.2, val_size = 0.2, batch_size=64, capacity=10000, epoches=10):
    train_loader, val_loader, test_loader = get_data_loaders(preprocessor, batch_size, capacity, test_size, val_size)
    X_train, y_train, X_val, y_val, X_test, y_test = get_corpus()
    model = None

    match model_name:
        case "mlp":
            model = MLPModel()
        case "regression":
            model = LogisticRegressionModel()
            model.fit(X_train, y_train)
            accuracy, recall, precision, f1, prob = get_regression_metrics(model, X_val, y_val, X_test, y_test)
            print(f"""    Accuracy = {accuracy}
                Recall = {recall}
                Precision = {precision}
                F1 = {f1}
                threshold = {prob}""")
            return
        case "rnn":
            vocabulary = get_vocabulary(X_train)
            model = RNNModel(len(vocabulary))
        case "lstm":
            vocabulary = get_vocabulary(X_train)
            model = LSTMModel(len(vocabulary))
        case _:
            raise NotImplementedError()

    trainer = Trainer(model, torch.optim.Adam(model.parameters()),
                          nn.BCEWithLogitsLoss(),
                          device=device)
    for epoch in range(EPOCHES):
        model.train()
        print(f"Loss: {trainer.train_epoch(train_loader)}")
        model.eval()
        threshold, f1_val, f1s = find_best_threshold(model, val_loader, device)
        f1_metric = BinaryF1Score(threshold=threshold).to(device)
        with torch.no_grad():
            f1_metric.reset()
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = torch.sigmoid(model(X_batch).reshape(-1))
                f1_metric.update(preds.reshape(-1), y_batch.reshape(-1))
        f1 = f1_metric.compute()
        print(f"""Epoch {epoch+1}
    Test F1 = {f1}
    Val F1 = {f1_val}
    Threshold = {threshold}""")
    torch.save(model.state_dict(), f"./{model_name}.pt")
