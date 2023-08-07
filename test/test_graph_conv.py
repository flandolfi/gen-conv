import pytest

import torch
from torch.nn import Conv1d, Conv2d

from gconv.conv import GenGraphConv
from benchmark.transforms import ImageToGraph, SequenceToGraph


@pytest.mark.parametrize('in_channels', [4, 8, 16])
@pytest.mark.parametrize('out_channels', [4, 8, 16])
@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('length', [3, 6, 9, 12, 15])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_conv_1d(in_channels, out_channels, groups, length, seed):
    torch.random.manual_seed(seed)
    rand_sequence = torch.rand(in_channels, length)

    conv = Conv1d(in_channels, out_channels,
                  kernel_size=3, stride=1,
                  padding='same', groups=groups)
    exp_out = conv(rand_sequence)

    graph_conv = GenGraphConv.from_regular_conv(conv, temperature='inf', similarity='neg-euclidean')
    data = SequenceToGraph()(rand_sequence)
    obs_out = graph_conv(x=data.x, edge_index=data.edge_index, pos=data.pos)

    assert torch.allclose(exp_out, obs_out.T, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize('in_channels', [4, 8, 16])
@pytest.mark.parametrize('out_channels', [4, 8, 16])
@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('width', [3, 6, 9])
@pytest.mark.parametrize('height', [3, 6, 9])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_conv_2d(in_channels, out_channels, groups, width, height, seed):
    torch.random.manual_seed(seed)
    transform = ImageToGraph()
    rand_image = torch.rand(in_channels, width, height)

    conv = Conv2d(in_channels, out_channels,
                  kernel_size=3, stride=1,
                  padding='same', groups=groups)
    out_image = conv(rand_image)
    exp_out = transform(out_image).x

    graph_conv = GenGraphConv.from_regular_conv(conv, temperature='inf', similarity='neg-euclidean')
    data = transform(rand_image)
    obs_out = graph_conv(x=data.x, edge_index=data.edge_index, pos=data.pos)

    assert torch.allclose(exp_out, obs_out, rtol=1e-4, atol=1e-7)
