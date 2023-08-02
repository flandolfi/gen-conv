import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


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
