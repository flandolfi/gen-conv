from abc import abstractmethod

import torch

from torch_geometric.data import InMemoryDataset

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


class Baseline(LightningModule):
    def __init__(self, dataset: InMemoryDataset,
                 lr: float = 0.001,
                 patience: int = 30):
        super(Baseline, self).__init__()

        self.dataset = dataset
        self.loss = torch.nn.CrossEntropyLoss()
        self.patience = patience
        self.lr = lr

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
        acc = self.accuracy(y_hat, data.y)
        self.log('test_acc', acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min",
                                   patience=self.patience)
        checkpoint = ModelCheckpoint(monitor="val_acc", mode="max")
        return [early_stop, checkpoint]
