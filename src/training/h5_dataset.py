import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler

from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

import anndata
import openslide


try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
class H5Dataset(Dataset):
    def __init__(self, input_filename, sep="\t", transforms=None, tokenizer=None):
        logging.debug(f'Loading h5 data from {input_filename}.')
        '''
        - input_filename is a path to a csv/txt file with list of h5 samples; for ex: HTAN-WUSTL_train.txt | HTAN-WUSTL_train.csv
        - The dataframe will give a list of all h5 files. We need to use these h5 files to get the openslide object and spot gene-expressions
        '''
        # fetch dataset_dir name from input_filename by splitting the extension and then "_"
        dataset_dir = os.path.splitext(input_filename)[0].split('_')[0]        
        h5_dir = os.path.join('datasets', dataset_dir, 'raw_visium')

        # read the dataframe
        df = pd.read_csv(os.path.join('datasets', dataset_dir, input_filename), sep=sep)
        h5_names = df['samples'].tolist()
        
        h5_dict = {i: h5_name.replace('\n', '') for i, h5_name in enumerate(h5_names)}

        # global variables
        self.h5_dir = h5_dir
        self.h5_dict = h5_dict
        self.transforms = transforms
        self.tokenize = tokenizer

        dataset, h5_adata_objects, h5_image_objects, spot_diameter_fullres = self.make_dataset(h5_dict)

        self.dataset = dataset
        self.h5_adata_objects = h5_adata_objects
        self.h5_image_objects = h5_image_objects
        self.spot_diameter_fullres = spot_diameter_fullres
        
        logging.debug('Done loading data.')

    def make_dataset(self, h5_dict):
        instances = []

        h5_image_objects = {}
        h5_adata_objects = {}
        spot_diameter_fullres = 0

        for (h5_index, h5_name) in h5_dict.items():

            h5_object = self.read_anndata(h5_name)   

            h5_image_path = h5_object.uns['spatial'][h5_name]['metadata']['source_image_path'] 
            h5_image_object = openslide.open_slide(h5_image_path)
            
            diameter = math.ceil(h5_object.uns['spatial'][h5_name]['scalefactors']['spot_diameter_fullres'])
            if spot_diameter_fullres < diameter:
                spot_diameter_fullres = diameter

            h5_adata_objects[h5_name] = h5_object
            h5_image_objects[h5_name] = h5_image_object
            
            spatial_coords = list(map(lambda coord: (tuple(coord), h5_index), h5_object.obsm['spatial']))
            instances.extend(spatial_coords)

        return instances, h5_adata_objects, h5_image_objects, spot_diameter_fullres

    def read_anndata(self, h5_name):

        if os.path.exists(os.path.join(self.h5_dir, h5_name)):
            h5_object = anndata.read_h5ad(os.path.join(self.h5_dir, h5_name, h5_name + '.h5ad'))
        else:
            raise Exception(f'{h5_name} does not exist.')
        
        return h5_object

    def read_instance(self, instance_coords, h5_index):

        # reads the h5 file from the h5_dict and outputs the corresponding openslide image and gene-expression
        h5_name = self.h5_dict[h5_index]
        # h5_object = self.read_anndata(h5_name)
        h5_object = self.h5_adata_objects[h5_name]

        # h5_image_name is saved in the h5 anndata object under uns['image_name']
        # h5_image_path = h5_object.uns['spatial'][h5_name]['metadata']['source_image_path']
        #h5_image_object is an efficient openslide object of the high resolution image
        # h5_image_object = openslide.open_slide(h5_image_path)
        h5_image_object = self.h5_image_objects[h5_name]

        # we get the spot_diameter_size from the spatial anndata file. This is the size of the spot in the image. We use this to get the center of the spot in the image
        # obtain spot_diameter_fullres from the h5 file and round it to the next integer
        image = h5_image_object.read_region(instance_coords, 0, (self.spot_diameter_fullres, self.spot_diameter_fullres)).convert('RGB')

        # h5_anndata_object is an efficient anndata object of the high resolution spatial file
        spot_idx = h5_object.obsm['spatial'].tolist().index(list(instance_coords))
        gexp = h5_object.X[spot_idx].todense()

        return image, gexp

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # instance_coords is a tuple of (x, y) coordinates of the spot in the image
        # h5_index is the index of the h5 file in the h5_dict
        image_coords, h5_index = self.dataset[idx]
        
        # read the patch surrounding the spot at instance_coords from the h5 file
        image, gexp = self.read_instance(image_coords, h5_index)
        if self.transforms:
            image = self.transforms(image)

        # get the gene-expression for the spot
        # tokenize is a QC measure to make sure the gene-expressions are valid and of fixed length
        if self.tokenize:
            gexp = self.tokenize(gexp)

        return image, gexp
    

if __name__ == "__main__":
    # transforms is just convert PIL to tensor
    transforms = transforms.Compose([transforms.ToTensor()])
    # tokenizer is just a function that converts the gene-expression to a torch tensor without normalizing
    tokenizer = lambda x: torch.tensor(x, dtype=torch.float32)
    dataset = H5Dataset('HTAN-WUSTL.csv', transforms=transforms, tokenizer=tokenizer)

    distributed = False
    sampler = DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0, pin_memory=True, sampler=sampler)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    data = DataInfo(dataloader, sampler)
    print(data)
    print(len(data.dataloader), len(dataset))
    for i, batch in enumerate(data.dataloader):
        print(batch[0].shape, batch[1].shape)

        if i > 20:
            break