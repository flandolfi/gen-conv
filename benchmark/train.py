from typing import Union, Optional
import warnings
import logging
import math
import glob
import os

from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import TUDataset, MalNetTiny
from torch_geometric import transforms as T

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from filelock import FileLock

from sklearn.model_selection import train_test_split

import pandas as pd

from tqdm import tqdm

from . import models
from .datasets import ModelNet40
from .transforms import PreSelect, ClonePos, RandomTranslate

warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_datasets(dataset: str = 'ModelNet40',
                 root: str = './data/',
                 valid_size: Union[float, int] = 0.1,
                 test_size: Union[float, int] = 0.2,
                 batch_size: int = -1,
                 num_workers: int = 0,
                 seed: int = 42):
    root = os.path.realpath(root)

    with FileLock(os.path.expanduser('~/.data.lock')):
        if dataset in {'ModelNet40', 'model-net'}:
            root = os.path.join(root, 'ModelNet40')
            pre_transform = T.Compose([
                PreSelect(1024),
                T.KNNGraph(20),
            ])
            train_transform = T.Compose([
                T.RandomScale((2/3, 3/2)),
                RandomTranslate(0.2),
                ClonePos(),
            ])
            valid_transform = ClonePos()

            train_dataset = ModelNet40(root=root, train=True,
                                       transform=train_transform,
                                       pre_transform=pre_transform)
            test_dataset = ModelNet40(root=root, train=False,
                                      transform=valid_transform,
                                      pre_transform=pre_transform)
            valid_dataset = None

            if valid_size > 0.:
                train_idx = list(range(len(train_dataset)))
                y = train_dataset.data.y.numpy()
                train_idx, valid_idx = train_test_split(train_idx,
                                                        test_size=valid_size,
                                                        random_state=seed,
                                                        stratify=y)
                valid_dataset = train_dataset[valid_idx]
                train_dataset = train_dataset[train_idx]
                valid_dataset.transform = valid_transform
        else:
            dataset = TUDataset(root=root, name=dataset)

            if dataset.num_node_features == 0:
                dataset.transform = T.Constant()

            idx = list(range(len(dataset)))
            y = dataset.data.y.numpy()

            train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                                   random_state=seed,
                                                   stratify=y)
            if valid_size > 0.:
                train_idx, valid_idx = train_test_split(train_idx, test_size=valid_size,
                                                        random_state=seed,
                                                        stratify=y[train_idx])
                valid_dataset = dataset[valid_idx]
            else:
                valid_dataset = None

            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]

    return LightningDataset(train_dataset, valid_dataset, test_dataset,
                            batch_size=batch_size, num_workers=num_workers)


def train(model: str = 'CustomDGCNN',
          dataset: str = 'ModelNet40',
          data_path: str = './data/',
          checkpoint_path: str = './models/{dataset}/{model}/',
          checkpoint_name: str = '{epoch}-{step}',
          valid_size: float = 0.1,
          test_size: float = 0.2,
          batch_size: int = 8,
          config: Optional[dict] = None,
          num_workers: int = 0,
          seed: int = 42,
          **trainer_kwargs):
    config = dict(config or {})
    checkpoint_kwargs = {
        'dirpath': checkpoint_path.format(dataset=dataset, model=model),
        'filename': checkpoint_name,
    }

    if valid_size > 0.:
        checkpoint_kwargs.update(monitor='val_acc', mode='max')
    
    pl.seed_everything(seed, workers=True)
    datamodule = get_datasets(dataset, data_path,
                              valid_size=valid_size,
                              test_size=test_size,
                              batch_size=batch_size,
                              num_workers=num_workers)
    model_cls = getattr(models, model)
    model = model_cls(dataset=datamodule.train_dataset, **config)
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(**checkpoint_kwargs)], 
                         **trainer_kwargs)
    trainer.fit(model, datamodule)  # noqa


def test(model: str = 'CustomDGCNN',
         dataset: str = 'ModelNet40',
         data_path: str = './data/',
         checkpoint_path: str = './models/{dataset}/{model}/',
         checkpoint_name: str = '*',
         test_size: float = 0.2,
         batch_size: int = 8,
         config: Optional[dict] = None,
         num_workers: int = 0,
         seed: int = 42,
         result_path: str = './results/',
         result_name: str = '{dataset}_{model}_test',
         **trainer_kwargs):
    config = dict(config or {})
    checkpoint_path = checkpoint_path.format(dataset=dataset, model=model)
    model_paths = glob.glob(os.path.join(checkpoint_path, checkpoint_name))
    result_name = result_name.format(dataset=dataset, model=model)

    if not result_name.endswith('.json'):
        result_name += '.json'

    pl.seed_everything(seed, workers=True)
    datamodule = get_datasets(dataset, data_path,
                              test_size=test_size,
                              batch_size=batch_size,
                              num_workers=num_workers)
    model_cls = getattr(models, model)
    model = model_cls(dataset=datamodule.train_dataset, **config)

    results = []

    for model_path in tqdm(model_paths):
        trainer = pl.Trainer(**trainer_kwargs)
        results.append(trainer.test(model, datamodule, ckpt_path=model_path)[0])

    os.makedirs(result_path, exist_ok=True)
    df_results = pd.DataFrame.from_records(results)
    results_path = os.path.join(result_path, result_name)
    df_results.to_json(results_path)

    logging.info(f"Model assessment results:\n\n"
                 f"{df_results}\n")
    logging.info(f"Results stored in {results_path}")
