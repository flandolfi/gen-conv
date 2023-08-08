import pytest

import torch
from torch.nn import Conv1d, Conv2d

from gconv.conv import GenPointConv
from benchmark.transforms import ImageToGraph, SequenceToGraph


@pytest.mark.parametrize('in_channels', [4, 8, 16])
@pytest.mark.parametrize('out_channels', [4, 8, 16])
@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
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

    graph_conv = GenPointConv.from_regular_conv(conv, temperature='inf')
    data = SequenceToGraph()(rand_sequence)
    obs_out = graph_conv(x=data.x, pos=data.pos)

    margin = kernel_size // 2
    assert torch.allclose(exp_out[:, margin:length - margin],
                          obs_out.T[:, margin:length - margin],
                          rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('in_channels', [4, 8, 16])
@pytest.mark.parametrize('out_channels', [4, 8, 16])
@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
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

    graph_conv = GenPointConv.from_regular_conv(conv, temperature='inf')
    data = transform(rand_image)
    obs_out = graph_conv(x=data.x, pos=data.pos)

    margin = kernel_size // 2
    mask = (data.pos > margin - 1) & (data.pos < data.pos.max(0, keepdims=True)[0] - margin + 1)
    mask = mask.all(-1)

    assert torch.allclose(exp_out[mask], obs_out[mask], rtol=1e-4, atol=1e-6)
