import torch.optim
from models.MLP.model import MLPModel
from torch import nn

from data.dataset import get_bow_data_loaders, get_tfidf_data_loaders
from metrics.metrics import get_metrics
from training.trainer import Trainer

MLP_model = MLPModel()

train_loader, test_loader = get_tfidf_data_loaders()

EPOCHES = 20

MLP_trainer = Trainer(MLP_model, torch.optim.Adam(MLP_model.parameters()),
                      nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.9])))

for epoch in range(EPOCHES):
    MLP_trainer.train_epoch(train_loader)
    accuracy, recall, precision, f1 = get_metrics(MLP_model, test_loader)
    print(f"""Epoch {epoch+1}
Accuracy = {accuracy}
Recall = {recall}
Precision = {precision}
F1 = {f1}""")

torch.save(MLP_model.state_dict(), "./mlp1.pt")

