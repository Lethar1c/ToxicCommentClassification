import torch


def loader_to_tensors(loader, device):
    xs = []
    ys = []

    for x, y in loader:
        xs.append(x.to(device))
        ys.append(y.to(device))

    X = torch.cat(xs)
    y = torch.cat(ys)

    return X, y