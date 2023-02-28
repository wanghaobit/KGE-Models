import torch
import torch.nn as nn
import torch.fft as fft
from torch_scatter import scatter


def scatter_(name, src, index, dim_size=None):
    if name == 'add':
        name = 'sum'
    assert name in ['sum', 'mean', 'max']
    out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
    return out[0] if isinstance(out, tuple) else out


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
    # return fft.irfft2(com_mult(conj(fft.rfft2(a, 1)), fft.rfft2(b, 1)), 1, signal_sizes=(a.shape[-1],))


def construct_adj(train_data, num_relations):
    edge_index, edge_type = [], []

    for sub, obj, rel in train_data:
        edge_index.append((sub, obj))
        edge_type.append(rel)

    # # Adding inverse edges
    # for sub, obj, rel in train_data:
    #     edge_index.append((obj, sub))
    #     edge_type.append(rel + num_relations)

    edge_index = torch.LongTensor(edge_index).cuda().t()
    edge_type = torch.LongTensor(edge_type).cuda()

    return edge_index, edge_type
