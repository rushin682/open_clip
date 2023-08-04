import logging
from collections import OrderedDict
from itertools import repeat
import collections.abc
from typing import Sequence, Callable, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import freeze_batch_norm_2d, _ntuple

from .ctranspath_swin_first import ConvStem
from .ctranspath_swin_first import swin_tiny_patch4_window7_224 as ctranspath_first
from .ctranspath_swin import swin_tiny_patch4_window7_224 as ctranspath

from .snn import SNN,  snn_small_omic, snn_big_omic
# from .scbert import scBERT
# from .scgpt import scGPT

class CustomImageModel(nn.Module):
    """custom image model adapter
    """

    def __init__(
            self,
            model_name,
            image_size=224,
            proj='identity',
            pretrained=False,
    ):
        super().__init__()
        self.image_size = _ntuple(2)(image_size)

        if model_name == 'ctranspath_first':
            self.trunk = ctranspath_first(embed_layer=ConvStem, pretrained=pretrained)
        
        elif model_name == 'ctranspath':
            self.trunk = ctranspath(embed_layer=ConvStem, pretrained=pretrained)
            
        if proj == 'identity':
            self.trunk.head = nn.Identity()

        if pretrained:
            td = torch.load('checkpoints/ctranspath.pth')
            # check all instances between td and self.trunk and load state dicts for only those that match
            for k, v in td['model'].items():
                if k in self.trunk.state_dict():
                    self.trunk.state_dict()[k].copy_(v)
                else:
                    raise KeyError(f'Unexpected key "{k}" in state_dict')

            print('Loaded pretrained weights from ctranspath.pth')
            # self.trunk[].load_state_dict(td['model'], strict=True)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)

            for ctranspath_first: unlock all layers
            for ctranspath: unlock last 2 layer groups or none
        """
        
        for param in self.trunk.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self.trunk)
        
        if unlocked_groups != 0:
        
            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True
            
            groups = [self.trunk.layers[-i] for i in range(1, unlocked_groups + 1)]
            _unlock(groups)
            _unlock(self.trunk.head)

    def forward(self, x):
        x = self.trunk(x)
        return x
    
class CustomTextModel(nn.Module):
    """custom image model adapter
    """
    def __init__(
            self,
            model_name,
            input_dim=512,
            output_dim=512,
            proj='identity',
            pretrained=False,            
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.context_length = output_dim

        if model_name == 'snn_small_omic':
            self.trunk = snn_small_omic(input_dim=input_dim, output_dim=output_dim)
            
        elif model_name == 'snn_big_omic':
            self.trunk = snn_big_omic(input_dim=input_dim, output_dim=output_dim)

        if proj == 'identity':
            self.trunk.head = nn.Identity()

        if pretrained:
            if model_name == 'snn_small_omic':
                td = torch.load('checkpoints/snn_small.pth')
            elif model_name == 'snn_big_omic':
                td = torch.load('checkpoints/snn_big.pth')

                
            # check all instances between td and self.trunk and load state dicts for only those that match
            for k, v in td['model'].items():
                if k in self.trunk.state_dict():
                    self.trunk.state_dict()[k].copy_(v)
                else:
                    raise KeyError(f'Unexpected key "{k}" in state_dict')

            print('Loaded pretrained weights from snn model')
    
    def forward(self, x):
        x = self.trunk(x)
        return x

# write a main function that calls the model and prints the output
def main():
    # instantiate the model
    model = CustomImageModel(model_name='swin_tiny_patch4_window7_224', image_size=224, proj='identity', pretrained=True)
    text_model = CustomTextModel(model_name='snn_small_omic', input_dim=600, output_dim=768, proj='identity', pretrained=False)
    # print the model
    print(text_model)
    # instantiate a random tensor
    x = torch.rand(1, 600)
    # pass the tensor through the model
    y = text_model(x)
    # print the output
    print(y)

if __name__ == '__main__':
    main()