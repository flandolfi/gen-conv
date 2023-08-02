from abc import abstractmethod

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch_geometric.data import InMemoryDataset

from pytorch_lightning import LightningModule

from ..metrics import accuracy, balanced_accuracy


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

        self.batch_y_true = []
        self.batch_y_pred = []

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
        acc = accuracy(y_hat.argmax(-1), data.y)
        
        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        self.log('val_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        
        return {
            'val_loss': loss,
            'val_acc': acc,
        }

    def compute_metrics(self):
        y_true = torch.cat(self.batch_y_true, dim=0)
        y_pred = torch.cat(self.batch_y_pred, dim=0)
        self.batch_y_pred.clear()
        self.batch_y_true.clear()

        return {
            'acc': accuracy(y_pred, y_true),
            'bal_acc': balanced_accuracy(y_pred, y_true),
        }

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.pos, data.edge_index, data.edge_attr, data.batch)

        self.batch_y_true.append(data.y.clone())
        self.batch_y_pred.append(y_hat.argmax(-1))

    def on_test_epoch_end(self):
        metrics = self.compute_metrics()

        for name, value in metrics.items():
            self.log('test_' + name, value, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)
        sch = CosineAnnealingWarmRestarts(opt, self.cosine_t_0, self.cosine_t_mult)
        return [opt], [sch]
