import torch
from torch import Tensor


def accuracy(y_pred: Tensor, y_true: Tensor):
    return torch.eq(y_pred, y_true).float().mean()


def balanced_accuracy(y_pred: Tensor, y_true: Tensor,
                      num_classes: int = None):
    if num_classes is None:
        num_classes = y_true.max() + 1
    
    eye = torch.eye(num_classes, dtype=torch.float, device=y_pred.device)
    y_pred, y_true = eye[y_pred], eye[y_true]

    conf_matrix = y_pred.T @ y_true
    return (torch.diagonal(conf_matrix)/conf_matrix.sum(0)).mean()

