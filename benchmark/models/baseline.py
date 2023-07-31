from abc import abstractmethod

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch_geometric.data import InMemoryDataset

from pytorch_lightning import LightningModule

from torchmetrics.functional import accuracy


class Baseline(LightningModule):
    def __init__(self, dataset: InMemoryDataset,
                 lr: float = 0.001,
                 cosine_t_0: int = 20,
                 cosine_t_mult: int = 2,
                 label_smoothing: float = 0.2):
        super(Baseline, self).__init__()

        self.dataset = dataset
        self.loss = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lr = lr
        self.cosine_t_0 = cosine_t_0
        self.cosine_t_mult = cosine_t_mult

    @staticmethod
    def accuracy(y_pred, y_true):
        y_class = torch.argmax(y_pred, dim=-1)
        return torch.mean(torch.eq(y_class, y_true).float())

    @abstractmethod
    def forward(self, x=None, pos=None, edge_index=None, edge_attr=None, batch=None):
        pass

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
        loss = self.loss(y_hat, data.y)
        self.log('train_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
        loss = self.loss(y_hat, data.y)
        acc = self.accuracy(y_hat, data.y)
        
        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)
        acc = accuracy(y_hat, data.y, task="multiclass", 
                       num_classes=self.dataset.num_classes)
        bal_acc = accuracy(y_hat, data.y, task="multiclass", average="macro", 
                           num_classes=self.dataset.num_classes)

        self.log('test_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('test_bal_acc', bal_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)
        sch = CosineAnnealingWarmRestarts(opt, self.cosine_t_0, self.cosine_t_mult)
        return [opt], [sch]
