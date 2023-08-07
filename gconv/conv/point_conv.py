from typing import Callable

import torch
from torch import Tensor

from torch_geometric.nn.pool import knn
from torch_geometric.typing import OptTensor

from .base_conv import BaseGenConv


class GenPointConv(BaseGenConv):
    def __init__(self, *args, k: int = 1, **kwargs):
        super(GenPointConv, self).__init__(*args, **kwargs)
        self.k = k

    def forward(self, x: Tensor, pos: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        if pos is None:
            pos = x

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        pts = pos.unsqueeze(-1) + self.offsets.T.unsqueeze(0)
        pts = pts.transpose(-1, -2).view(-1, self.pos_channels)
        pts_batch = batch.repeat_interleave(self.num_offsets)

        pts_idx, pos_idx = knn(x=pos, y=pts, k=self.k, batch_x=batch, batch_y=pts_batch)

        dist = torch.norm(pts[pts_idx] - pos[pos_idx], p=2, dim=-1).view(-1, self.k)
        sim = torch.softmax(dist * self.temperature, dim=-1)

        x_j = torch.einsum('pk,pkc->pc', sim, x[pos_idx].view(-1, self.k, self.in_channels))
        x_j = x_j.view(-1, self.num_offsets, self.groups, self.in_channels // self.groups)

        W = self.weights.view(self.num_offsets, self.groups,
                              self.out_channels // self.groups,
                              self.in_channels // self.groups)
        out = torch.einsum('ogji,pogi->pj', W, x_j)

        if self.bias is not None:
            out = out + self.bias

        return out
