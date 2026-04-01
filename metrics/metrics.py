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
    try:
        model.eval()
    except:
        pass
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
        for prob in tqdm(probs):
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

        is_first_batch = True

        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device).int()
            y_pred = torch.sigmoid(model(x))
            y_pred_01 = (y_pred > max_prob).int()
            tp += ((y == 1) & (y_pred_01 == 1)).sum()
            tn += ((y == 0) & (y_pred_01 == 0)).sum()
            fp += ((y == 0) & (y_pred_01 == 1)).sum()
            fn += ((y == 1) & (y_pred_01 == 0)).sum()
            if is_first_batch:
                is_first_batch = False
                print(f"""
Model predictions:
y_pred = {y_pred}
Current batch info:
x_mean - {x.mean()}
x_std - {x.std()}""")
                # for i, param in enumerate(model.parameters()):
                #     if isinstance(param, torch.Tensor):
                #         print(f"Param {i} -- mean = {param.mean()}, std = {param.std()}, max = {param.max()}, min = {param.min()}")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (fp + fn + tp + tn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1, max_prob

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_regression_metrics(regression, x_val, y_val, x_test, y_test, prob_count=200):
    '''
    Get model metrics (accuracy, recall, precision, F1)
    :param model:
    :param val_loader:
    :param test_loader:
    :param alpha: - порог для весов
    :return: ``(accuracy,recall,precision,F1)``
    '''
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    max_prob = 0
    probs = np.linspace(1e-4, 1-1e-4, prob_count)
    y_pred = regression.predict_proba(x_val)[:, 1]
    max_f1 = -1
    for prob in tqdm(probs):
        y_pred_01_val = (y_pred > prob).astype(int)
        val_tp = ((y_val == 1) & (y_pred_01_val == 1)).sum()
        val_tn = ((y_val == 0) & (y_pred_01_val == 0)).sum()
        val_fp = ((y_val == 0) & (y_pred_01_val == 1)).sum()
        val_fn = ((y_val == 1) & (y_pred_01_val == 0)).sum()
        val_precision = val_tp / (val_tp + val_fp + 1e-9)
        val_recall = val_tp / (val_tp + val_fn + 1e-9)
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-9)
        if val_f1 > max_f1:
            max_f1 = val_f1
            max_prob = prob

    y_pred = regression.predict_proba(x_test)[:, 1]
    y_pred_01 = (y_pred > max_prob).astype(int)
    tp += ((y_test == 1) & (y_pred_01 == 1)).sum()
    tn += ((y_test == 0) & (y_pred_01 == 0)).sum()
    fp += ((y_test == 0) & (y_pred_01 == 1)).sum()
    fn += ((y_test == 1) & (y_pred_01 == 0)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (fp + fn + tp + tn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return accuracy, recall, precision, f1, max_prob