import numpy as np
from tqdm import tqdm
from torchmetrics.classification import BinaryF1Score

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_regression_metrics(regression, x_val, y_val, x_test, y_test, prob_count=200):
    """
    :param regression:
    :param x_val:
    :param y_val:
    :param x_test:
    :param y_test:
    :param prob_count:
    :return:
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    max_prob = 0
    probs = np.linspace(1e-4, 1-1e-4, prob_count)
    y_pred = regression.predict_proba(x_val)
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

    y_pred = regression.predict_proba(x_test)
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


def find_best_threshold(model, val_loader, device):
    thresholds = np.linspace(0.2, 0.8, 61)
    best_threshold = 0
    best_f1 = 0

    f1s = []

    for t in tqdm(thresholds):
        f1_score = BinaryF1Score(threshold=float(t)).to(device)
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            f1_score.update(model(x).reshape(-1).to(device), y.reshape(-1).to(device))
        if (score := f1_score.compute()) > best_f1:
            best_threshold = t
            best_f1 = score
        f1s.append(score)

    return best_threshold, best_f1, f1s


