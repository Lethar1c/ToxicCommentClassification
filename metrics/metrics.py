import numpy as np
import torch
from tqdm import tqdm

from features.loader_to_tensors import loader_to_tensors


def get_metrics(model, val_loader, test_loader, device, prob_count=200):
    '''
    Get model metrics (accuracy, recall, precision, F1)
    :param model:
    :param val_loader:
    :param test_loader:
    :param alpha: - порог для весов
    :return: ``(accuracy,recall,precision,F1)``
    '''
    model.eval()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    max_prob = 0
    with torch.no_grad():
        probs = np.linspace(1e-4, 1-1e-4, prob_count)
        x_val, y_val = loader_to_tensors(val_loader, device=device)
        y_pred = torch.sigmoid(model(x_val))
        max_f1 = -1
        for prob in probs:
            y_pred_01_val = (y_pred > prob).int()
            val_tp = ((y_val == 1) & (y_pred_01_val == 1)).sum()
            val_tn = ((y_val == 0) & (y_pred_01_val == 0)).sum()
            val_fp = ((y_val == 0) & (y_pred_01_val == 1)).sum()
            val_fn = ((y_val == 1) & (y_pred_01_val == 0)).sum()
            val_precision = val_tp / (val_tp + val_fp)
            val_recall = val_tp / (val_tp + val_fn)
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-9)
            if val_f1 > max_f1:
                max_f1 = val_f1
                max_prob = prob

        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device).int()
            y_pred = torch.sigmoid(model(x))
            y_pred_01 = (y_pred > max_prob).int()
            tp += ((y == 1) & (y_pred_01 == 1)).sum()
            tn += ((y == 0) & (y_pred_01 == 0)).sum()
            fp += ((y == 0) & (y_pred_01 == 1)).sum()
            fn += ((y == 1) & (y_pred_01 == 0)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (fp + fn + tp + tn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1, max_prob