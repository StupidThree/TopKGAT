import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import args
import numpy as np


def sparse_sum(values, indices0, indices1, n):
    if indices0 is None:
        assert (len(indices1.shape) == 1 and values.shape[0] == indices1.shape[0])
    else:
        assert (len(indices0.shape) == 1 and len(indices1.shape) == 1)
        assert (indices0.shape[0] == indices1.shape[0])
    # assert (len(values.shape) <= 2)
    return torch.zeros([n]+list(values.shape)[1:], device=values.device, dtype=values.dtype).index_add(0, indices1, values if indices0 is None else values[indices0])


def rest_sum(values, indices0, indices1, n):
    return values.sum(0).unsqueeze(0)-sparse_sum(values, indices0, indices1, n)


class TopKformer(nn.Module):
    def __init__(self, dataset):
        super(TopKformer, self).__init__()
        self.dataset = dataset
        self.omega = lambda x: 4/(1+(-x).exp())/(1+x.exp())
        self.n, self.m = self.dataset.num_users, self.dataset.num_items
        self.u, self.i = self.dataset.train_user, self.dataset.train_item
        self.du, self.di = self.dataset.du.unsqueeze(1), self.dataset.di.unsqueeze(1)

    def forward(self, x, beta):
        xu, xi = torch.split(x, [self.n, self.m])
        zu, zi = torch.split(F.normalize(x), [self.n, self.m])
        omega_value = self.omega(((zu[self.u]*zi[self.i]).sum(1)-beta[self.u]).unsqueeze(1))/self.du[self.u].sqrt()/self.di[self.i].sqrt()
        dx = torch.concat([
            sparse_sum(omega_value*xi[self.i], None, self.u, self.n),
            sparse_sum(omega_value*xu[self.u], None, self.i, self.m)], dim=0)
        return dx
