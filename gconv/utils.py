import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch_geometric.nn.pool import radius_graph as radius_graph_p2
from torch_geometric.typing import Adj, OptTensor, SparseTensor, Tensor
from torch_geometric.utils import scatter, to_torch_coo_tensor, to_edge_index


def maximal_independent_set(edge_index: Adj, k: int = 1,
                            num_nodes: Optional[int] = None,
                            perm: OptTensor = None) -> Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a :class:`ByteTensor`) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.

    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.

    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).

    :rtype: :class:`ByteTensor`
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = num_nodes

        if n is None:
            n = edge_index.max().item() + 1

    # TODO: Use scatter's `out` and `include_self` arguments,
    #       when available, instead of adding self-loops
    self_loops = torch.arange(n, dtype=torch.long, device=device)
    row, col = torch.cat([row, self_loops]), torch.cat([col, self_loops])

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(k):
            min_rank = scatter(min_rank[col], row, dim_size=n, reduce='min')

        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()

        for _ in range(k):
            mask = scatter(mask[row], col, dim_size=n, reduce='max')

        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis


def k_hop(edge_index: Adj, edge_attr: OptTensor = None, k: int = 1,
          num_nodes: Optional[int] = None) -> Tuple[Adj, OptTensor]:
    if torch.is_tensor(edge_index):
        adj = to_torch_coo_tensor(edge_index, edge_attr, num_nodes)
    else:
        adj = edge_index

    out = adj

    for _ in range(1, k):
        out = adj @ out

    if torch.is_tensor(edge_index):
        return to_edge_index(out)

    return out, None


def aggregate_k_hop(x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                    k: int = 1, reduce: str = 'mean') -> Tensor:
    k_hop_index, _ = k_hop(edge_index, edge_attr, k)

    if not torch.is_tensor(k_hop_index):
        k_hop_index, _ = to_edge_index(k_hop_index)

    row, col = k_hop_index
    return scatter(x[row], col, dim_size=x.size(0), reduce=reduce)


def radius_graph(x: Tensor, r: float, p: float = 2, batch: OptTensor = None,
                 loop: bool = True, max_num_neighbors: int = 32,
                 flow: str = 'source_to_target',
                 num_workers: int = 1) -> Tensor:
    if math.isinf(p):
        adjusted_r = r*math.sqrt(2)
    elif p > 2:
        adjusted_r = r*math.sqrt(2)/math.pow(2, 1/p)
    else:
        adjusted_r = r

    edge_index = radius_graph_p2(x, adjusted_r, batch, loop,
                                 max_num_neighbors, flow, num_workers)

    row, col = edge_index
    dist = F.pairwise_distance(x[row], x[col], p=p, eps=0.)
    mask = dist <= r

    return edge_index[:, mask]
