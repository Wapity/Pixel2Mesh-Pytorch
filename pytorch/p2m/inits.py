import torch
import torch.nn as nn


def create_variable(size, initiliaze=False):
    variable = nn.Parameter(torch.zeros(size, dtype=torch.float))
    nn.init.xavier_uniform_(variable.data)
    return variable


def uniform(shape, scale=0.05):
    """Uniform init."""
    variable = nn.Parameter(torch.zeros(shape, dtype=torch.float))
    nn.init.uniform_(variable.data, a=-scale, b=scale)
    return variable


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    variable = nn.Parameter(torch.zeros(shape, dtype=torch.float))
    nn.init.xavier_uniform_(variable.data)
    return variable


def zeros(shape):
    """All zeros."""
    variable = nn.Parameter(torch.zeros(shape, dtype=torch.float))
    return variable


def ones(shape, name=None):
    """All ones."""
    variable = nn.Parameter(torch.ones(shape, dtype=torch.float))
    return variable
