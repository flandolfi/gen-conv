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


class ClonePos(BaseTransform):
    def __call__(self, data: Data):
        data['x'] = data.pos
        return data
