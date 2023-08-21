import os.path as osp
import glob
import h5py
from typing import List, Tuple, Union

import torch
from torchvision.datasets import ImageNet
from torchvision.datasets.imagenet import ARCHIVE_META, META_FILE

from torch_geometric.data import (
    InMemoryDataset,
    Dataset,
    download_url,
    extract_zip,
    Data
)

from benchmark.transforms import ImageToGraph


class ModelNet40(InMemoryDataset):
    url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(ModelNet40, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['modelnet40_ply_hdf5_2048.zip']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        data_list = []

        for h5_name in glob.glob(osp.join(self.raw_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % dataset)):
            f = h5py.File(h5_name)
            pos_part = f['data'][:].astype('float32')
            y_part = f['label'][:].astype('int64')
            f.close()

            for pos, y in zip(pos_part, y_part):
                data_list.append(Data(pos=torch.as_tensor(pos),
                                      y=torch.as_tensor(y)))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


class GraphImageNet(Dataset):
    def __init__(self, root, split='train', image_transform=None, graph_transform=None):
        self.image_net = ImageNet(root, split, transform=image_transform)
        self.image_to_graph = ImageToGraph()

        super().__init__(root, transform=graph_transform)

    def get(self, idx: int) -> Data:
        sample, target = self.image_net[idx]
        data = self.image_to_graph(sample)
        data.y = torch.tensor([target], dtype=torch.long)
        return data

    def len(self) -> int:
        return len(self.image_net)

    def download(self):
        pass

    def process(self):
        pass

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return self.root

    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [name for name, md5 in ARCHIVE_META.items()]

    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return META_FILE
