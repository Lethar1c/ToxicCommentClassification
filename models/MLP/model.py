import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, capacity=10000):
        super().__init__()
        self.input = nn.Linear(capacity, 512)
        self.hidden1 = nn.Linear(512, 256)
        self.hidden2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)

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
