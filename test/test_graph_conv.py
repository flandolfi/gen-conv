import pytest

import torch
from torch.nn import Conv1d, Conv2d

from gconv.conv import GenGraphConv
from gconv.utils import k_hop
from benchmark.transforms import ImageToGraph, SequenceToGraph


@pytest.mark.parametrize('in_channels', [4, 8, 16])
@pytest.mark.parametrize('out_channels', [4, 8, 16])
@pytest.mark.parametrize('kernel_size', [3, 5, 7, 11])
@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('length', [8, 16, 32])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_conv_1d(in_channels, out_channels, kernel_size, groups, length, seed):
    torch.random.manual_seed(seed)
    rand_sequence = torch.rand(in_channels, length)

    conv = Conv1d(in_channels, out_channels,
                  kernel_size=kernel_size, stride=1,
                  padding='same', groups=groups)
    exp_out = conv(rand_sequence)

    graph_conv = GenGraphConv.from_regular_conv(conv, temperature='inf', similarity='neg-euclidean')
    data = SequenceToGraph()(rand_sequence)
    k_hop_index, _ = k_hop(data.edge_index, k=kernel_size//2)
    obs_out = graph_conv(x=data.x, edge_index=k_hop_index, pos=data.pos)

    assert torch.allclose(exp_out, obs_out.T, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('in_channels', [4, 8, 16])
@pytest.mark.parametrize('out_channels', [4, 8, 16])
@pytest.mark.parametrize('kernel_size', [3, 5, 7, 11])
@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('width', [8, 16, 32])
@pytest.mark.parametrize('height', [8, 16, 32])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_conv_2d(in_channels, out_channels, kernel_size, groups, width, height, seed):
    torch.random.manual_seed(seed)
    transform = ImageToGraph()
    rand_image = torch.rand(in_channels, width, height)

    conv = Conv2d(in_channels, out_channels,
                  kernel_size=kernel_size, stride=1,
                  padding='same', groups=groups)
    out_image = conv(rand_image)
    exp_out = transform(out_image).x

    graph_conv = GenGraphConv.from_regular_conv(conv, temperature='inf', similarity='neg-euclidean')
    data = transform(rand_image)
    k_hop_index, _ = k_hop(data.edge_index, k=kernel_size//2)
    obs_out = graph_conv(x=data.x, edge_index=k_hop_index, pos=data.pos)

    assert torch.allclose(exp_out, obs_out, rtol=1e-4, atol=1e-6)
