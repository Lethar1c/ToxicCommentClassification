import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for x, y in tqdm(dataloader):
            # print(x)
            # x, y = x.to(self.device), y.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y.float().reshape(-1, 1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)
