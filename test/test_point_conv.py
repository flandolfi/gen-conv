import pytest

import torch
from torch.nn import Conv1d, Conv2d

from gconv.conv import GenPointConv
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

    graph_conv = GenPointConv.from_regular_conv(conv, temperature='inf')
    data = SequenceToGraph()(rand_sequence)
    obs_out = graph_conv(x=data.x, pos=data.pos)

    assert torch.allclose(exp_out[:, 1:-1], obs_out.T[:, 1:-1], rtol=1e-4, atol=1e-7)


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

    graph_conv = GenPointConv.from_regular_conv(conv, temperature='inf')
    data = transform(rand_image)
    obs_out = graph_conv(x=data.x, pos=data.pos)

    mask = (data.pos > 0) & (data.pos < data.pos.max(0, keepdims=True)[0])
    mask = mask.all(-1)

    assert torch.allclose(exp_out[mask], obs_out[mask], rtol=1e-4, atol=1e-7)
