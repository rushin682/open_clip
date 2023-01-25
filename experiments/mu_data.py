import os
import json
import math
import h5py
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

import openslide
import anndata
from anndata import AnnData
def load_json(json_file):

    with open(json_file) as f:
        gene_sets = json.load(f)
    
    return gene_sets

def find_slides(directory):
    """Finds the slides to be used for dataset creation

    Write description of default function
    """
    process_list = os.path.join(directory, 'process_list_autogen.csv')
    if os.path.isfile(process_list):
        slides = list(pd.read_csv(process_list)['slide_id'])
    else:
        if os.path.join(directory, 'WSIs').is_dir():
            slides = os.listdir(os.path.join(directory, 'WSIs'))


    if not slides:
        raise FileNotFoundError(f"Couldn't find any processed slides in {directory}.")

    slide_to_idx = {i: slide_name for i, slide_name in enumerate(slides)}
    return slides, slide_to_idx

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)
    
def make_dataset(directory,
                slide_to_idx = None,
                extensions = None,
                is_valid_file = None):
    """Generates a list of samples of a form (path_to_sample, slide).

    See :  Write description of default function.
    """
    directory = os.path.expanduser(directory)

    # Should in general not occur
    if slide_to_idx is None:
        _, slide_to_idx = find_slides(directory)
    elif not slide_to_idx:
        raise ValueError("'slide_to_idx' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_slides = set()
    for (target_slide_index, target_slide) in slide_to_idx.items():
        target_slide_patches = target_slide[:-4]+'.h5'
        h5_slide_file = os.path.join(directory, 'patches', target_slide_patches)
        if is_valid_file(h5_slide_file):
            with h5py.File(h5_slide_file, "r") as f:
                patch_level = f['coords'].attrs['patch_level']
                patch_size = f['coords'].attrs['patch_size']
                for coord in f['coords']:
                    item = coord, target_slide_index
                    instances.append(item)

                    if target_slide not in available_slides:
                        available_slides.add(target_slide)


    empty_slides = set(slide_to_idx.values()) - available_slides
    if empty_slides:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_slides))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances, patch_level, patch_size


class MuDataset(Dataset):
    def __init__(self, root=None, ids=None, gene_set_list='auto'):
        super(MuDataset, self).__init__()

        if gene_set_list == 'auto':
            # anndata files ALWAYS contain all genes. So these are just some subsets
            self.gene_set_list = ['HALLMARK_KRAS_SIGNALING_UP', 
                                  'HALLMARK_TGF_BETA_SIGNALING', 
                                  'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION', 
                                  'HALLMARK_INTERFERON_GAMMA_RESPONSE', 
                                  'HALLMARK_ANGIOGENESIS', 
                                  'HALLMARK_INFLAMMATORY_RESPONSE', 
                                  'HALLMARK_HYPOXIA', 
                                  'HALLMARK_G2M_CHECKPOINT', 
                                  'HALLMARK_APOPTOSIS', 
                                  'HALLMARK_TNFA_SIGNALING_VIA_NFKB']

        elif type(gene_set_list) == list:
            self.gene_set_list = gene_set_list 

        self.gene_sets = load_json(os.path.join(root, "data", "hallmark_genesets.json"))
            
        self.transform = transforms.Compose([transforms.ToTensor()])


    @staticmethod
    def make_dataset(directory,
                     slide_to_idx,
                     extensions=None,
                     is_valid_file=None):

        """
        Write your description here
        """
        if slide_to_idx is None:
            # prevent potential bug since make_dataset() would use the slide_to_idx logic of the
            # find_slides() function, instead of using that of the find_slides() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The slide_to_idx parameter cannot be None.")

        return make_dataset(directory, slide_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    @staticmethod
    def find_slides(self, directory):
        """
        Write your description here
        """

        return find_slides(directory)
            
    @abstractmethod
    def __getitem__(self, index): # maybe call this function from the inheriting function for common functionality?
        """
        Write your description here
        Parameters:
            index (int): Index of the slide from a slide_list given by 'ids'
        """

        pass

    def __len__(self):
        return len(self.ids)