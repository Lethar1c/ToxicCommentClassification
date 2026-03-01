import torch

from data.dataset import get_tfidf_data_loaders
from models.MLP.model import MLPModel

MLP_model = MLPModel()

train_loader, val_loader, test_loader = get_tfidf_data_loaders()

device = "cuda" if torch.cuda.is_available() else 'cpu'
print("Running on " + device)

for x, y in train_loader:
    print(x)
    print(y)
    print(MLP_model(x))
    break
