from pathlib import Path

import joblib
import torch.optim
from torch.utils.data import DataLoader

from features.loader_to_tensors import loader_to_tensors
from features.tokenizer import Vocabulary
from models.LogisticRegression.model import LogisticRegressionModel
from models.MLP.model import MLPModel
from torch import nn
from torchmetrics.functional import accuracy, f1_score, recall, precision

from data.dataset import get_bow_data_loaders, get_tfidf_data_loaders, get_corpus, CommentDataset, get_rnn_data_loaders, \
    get_rnn_corpus
from metrics.metrics import get_metrics, get_regression_metrics, find_best_threshold
from models.RNN.model import RNNModel
from training.trainer import Trainer
from torchmetrics.classification import BinaryF1Score

# MLP_model = MLPModel()
#
# train_loader, val_loader, test_loader = get_tfidf_data_loaders()
#
EPOCHES = 10
#
device = "cuda" if torch.cuda.is_available() else 'cpu'

VOCABULARY_PATH = Path("saves") / "vocabulary.pkl"

# print("Running on " + device)
#
# MLP_trainer = Trainer(MLP_model, torch.optim.Adam(MLP_model.parameters()),
#                       nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.9], device=device)),
#                       device=device)
#
#
# for epoch in range(EPOCHES):
#     MLP_trainer.train_epoch(train_loader)
#     probs = torch.linspace(0.005, 0.99, 200)
#     accuracy, recall, precision, f1, prob = get_metrics(MLP_model, val_loader, test_loader, device=device)
#
#     print(f"""Epoch {epoch+1}
# Accuracy = {accuracy}
# Recall = {recall}
# Precision = {precision}
# F1 = {f1}
# threshold = {prob}""")
#
# torch.save(MLP_model.state_dict(), "./mlp1.pt")

def train_regression():
    regression = LogisticRegressionModel()
    X_train, y_train, X_val, y_val, X_test, y_test = get_corpus()
    regression.fit(X_train, y_train)

    val_loader = DataLoader(CommentDataset(X_val, y_val))
    test_loader = DataLoader(CommentDataset(X_test, y_test))

    accuracy, recall, precision, f1, prob = get_regression_metrics(regression, X_val, y_val, X_test, y_test)
    print(f"""    Accuracy = {accuracy}
    Recall = {recall}
    Precision = {precision}
    F1 = {f1}
    threshold = {prob}""")

# train_regression()

def train_rnn():
    print("getting data loaders")
    train_loader, val_loader, test_loader = get_rnn_data_loaders()
    X_train, y_train, X_val, y_val, X_test, y_test = get_corpus()

    # X_test_m, y_test_m = get_rnn_corpus(device)

    try:
        print("loading vocabulary")
        vocabulary = joblib.load(VOCABULARY_PATH)
    except Exception:
        print("initializing vocabulary")
        vocabulary = Vocabulary()
        vocabulary.build(X_train)
        print("vocabulary initialized. saving...")
        joblib.dump(vocabulary, VOCABULARY_PATH)

    print("creating model")
    rnn = RNNModel(len(vocabulary))
    rnn = rnn.to(device)

    print("Running RNN on " + device)

    RNN_trainer = Trainer(rnn, torch.optim.Adam(rnn.parameters()),
                          nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.9], device=device)),
                          device=device)


    print("Start training")

    # f1_metric = BinaryF1Score(threshold=0.3).to(device)

    for epoch in range(EPOCHES):
        rnn.train()
        print(f"Loss: {RNN_trainer.train_epoch(train_loader)}")
        # torch.save(RNN_trainer.model.state_dict(), Path("saves") / "goida")
        # probs = torch.linspace(0.005, 0.99, 200)
        # accuracy, recall, precision, f1, prob = get_metrics(rnn, val_loader, test_loader, device=device)
        # accuracy, recall, precision, f1, prob = get_metrics(rnn, train_loader, train_loader, device=device)

        # acc = accuracy(rnn(X_test_m), y_test_m, "binary", 0.3)
        # rec = recall(rnn(X_test_m), y_test_m, "binary", 0.3)
        # pre = precision(rnn(X_test_m), y_test_m, "binary", 0.3)
        # f1 = f1_score(rnn(X_test_m), y_test_m, "binary", 0.3)
        rnn.eval()

        threshold, f1_val = find_best_threshold(rnn, val_loader, device)

        f1_metric = BinaryF1Score(threshold=threshold).to(device)

        with torch.no_grad():
            f1_metric.reset()
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = torch.sigmoid(rnn(X_batch).reshape(-1))
                # print(preds)

                f1_metric.update(preds.reshape(-1), y_batch.reshape(-1))

                # accs.append(accuracy(preds.reshape(-1), y_batch, "binary", 0.3))
                # recs.append(recall(preds.reshape(-1), y_batch, "binary", 0.3))
                # pres.append(precision(preds.reshape(-1), y_batch, "binary", 0.3))
                # f1s.append(f1_score(preds.reshape(-1), y_batch, "binary", 0.3))

        # acc = sum(accs) / len(accs)
        # rec = sum(recs) / len(recs)
        # pre = sum(pres) / len(pres)
        f1 = f1_metric.compute()

        print(f"""Epoch {epoch+1}
    Test F1 = {f1}
    Val F1 = {f1_val}""")
    #     print(f"""Epoch {epoch+1}
    # Accuracy = {acc}
    # Recall = {rec}
    # Precision = {pre}
    # F1 = {f1}""")

    torch.save(rnn.state_dict(), "./rnn1.pt")

train_rnn()



