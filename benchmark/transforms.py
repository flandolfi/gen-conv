from typing import Union

import torch

from PIL.Image import Image

from torchvision.transforms import functional as F

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import grid, to_undirected


class PreSelect(BaseTransform):
    def __init__(self, num_nodes: int = 1024):
        self.num_nodes = num_nodes

    def __call__(self, data: Data):
        if data.x is not None:
            data['x'] = data.x[:self.num_nodes]

        if data.pos is not None:
            data['pos'] = data.pos[:self.num_nodes]

        data['num_nodes'] = self.num_nodes
        return data


class RandomTranslate(BaseTransform):
    def __init__(self, delta_max: float = 0.2):
        self.delta_max = delta_max
    
    def __call__(self, data: Data):
        pos = data.pos
        translation = torch.rand((1, pos.size(-1)), 
                                 dtype=pos.dtype, device=pos.device)
        data['pos'] = pos + 2*self.delta_max*translation - self.delta_max
        return data


class ClonePos(BaseTransform):
    def __call__(self, data: Data):
        data['x'] = data.pos
        return data


class GetPosInfo(BaseTransform):
    def __init__(self, index: slice):
        self.index = index

    def __call__(self, data: Data):
        data['pos'] = data.x[:, self.index]
        return data


class ImageToGraph(BaseTransform):
    def __call__(self, data: Union[Image, torch.Tensor]):
        if not torch.is_tensor(data):
            data = F.to_tensor(data)

        edge_index, pos = grid(data.shape[2], data.shape[1])
        x = data.permute(1, 2, 0)[tuple(pos.T.long())]

        return Data(x=x, edge_index=edge_index, pos=pos)


class SequenceToGraph(BaseTransform):
    def __call__(self, data: torch.Tensor):
        length = data.size(-1)
        pos = torch.arange(length, dtype=torch.float).view(-1, 1)

        idx = torch.arange(length * 2, dtype=torch.long)
        row = idx // 2
        col = (idx % 2) + row
        mask = col < length
        row, col = row[mask], col[mask]
        edge_index = to_undirected(edge_index=torch.stack([row, col]),
                                   num_nodes=length)

        return Data(x=data.T, edge_index=edge_index, pos=pos)
