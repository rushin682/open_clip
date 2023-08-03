import logging
from collections import OrderedDict
from itertools import repeat
import collections.abc
from typing import Sequence, Callable, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

class SNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, model_size: str='small', n_classes: int=0):
        super().__init__()
        self.n_classes = n_classes
        self.size_dict = {'small': [256, 256, 256, output_dim], 'big': [1024, 1024, 1024, output_dim]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict[model_size]
        fc_layers = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_layers.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_layers = nn.Sequential(*fc_layers)
        self.head = nn.Linear(hidden[-1], n_classes) if n_classes > 0 else nn.Identity()
        init_max_weights(self)

    def forward(self, x):

        features = self.fc_layers(x)
        logits = self.head(features).unsqueeze(0)

        return logits

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

def snn_small_omic(input_dim: int, output_dim = 768, n_classes: int=0):
    return SNN(input_dim=input_dim, output_dim=output_dim, model_size='small', n_classes=n_classes)

def snn_big_omic(input_dim: int, output_dim = 768, n_classes: int=0):
    return SNN(input_dim=input_dim, output_dim=output_dim, model_size='big', n_classes=n_classes)