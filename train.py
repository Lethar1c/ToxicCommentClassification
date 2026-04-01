from pathlib import Path

import torch.optim
from torch.utils.data import DataLoader

from models.LogisticRegression.model import LogisticRegressionModel
from models.MLP.model import MLPModel
from torch import nn

from data.dataset import get_bow_data_loaders, get_tfidf_data_loaders, get_corpus, CommentDataset
from metrics.metrics import get_metrics, get_regression_metrics
from training.trainer import Trainer

# MLP_model = MLPModel()
#
# train_loader, val_loader, test_loader = get_tfidf_data_loaders()
#
# EPOCHES = 20
#
device = "cuda" if torch.cuda.is_available() else 'cpu'
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
train_regression()

