import torch
from tqdm import tqdm

def get_metrics(model, test_loader):
    '''
    Get model metrics (accuracy, recall, precision, F1)
    :param model:
    :param test_loader:
    :return: ``(accuracy,recall,precision,F1)``
    '''
    model.eval()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            y = y.int()
            y_pred = torch.sigmoid(model(x))
            y_pred_01 = (y_pred > 0.3).int()
            tp += ((y == 1) & (y_pred_01 == 1)).sum()
            tn += ((y == 0) & (y_pred_01 == 0)).sum()
            fp += ((y == 0) & (y_pred_01 == 1)).sum()
            fn += ((y == 1) & (y_pred_01 == 0)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (fp + fn + tp + tn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1