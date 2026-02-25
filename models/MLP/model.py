import torch.nn as nn
from torch.nn import Linear
import torch

class MLPModel(nn.Module):
    def __init__(self, capacity=10000):
        super().__init__()
        self.input = Linear(capacity, 512)
        self.hidden1 = Linear(512, 256)
        self.hidden2 = Linear(256, 64)
        self.output = Linear(64, 1)

    def forward(self, x):
        h1 = torch.relu(self.input(x))
        h2 = torch.relu(self.hidden1(h1))
        h3 = torch.relu(self.hidden2(h2))
        return self.output(h3)

    # def compute_loss(self, y_pred, y):
    #     return ((y_pred - y) ** 2).sum()


# model = MLPModel()
#
# print(model("hello nigger"))
# print(model("you are the kindest person ever!"))
