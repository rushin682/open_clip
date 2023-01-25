import os
import json
import logging
import math
import random
import sys

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold

import torch
import torchvision.datasets as datasets
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from her2st_data import Her2stDataSet 

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def _convert_to_rgb(image):
    return image.convert('RGB')

def read_file(file_name):
    with open(file_name, 'r') as f:
        records = list(f)

    return records

def separate_data(ids, seed, n_folds=1, fold_idx=0):
    assert 0 <= fold_idx and fold_idx < n_folds, "fold_idx must be from 0 to 9."
    # sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.8, random_state=seed)
    k = KFold(n_splits=n_folds)

    print("Total Train-Valid size: 0.8:0.2")

    idx_list = []
    for idx in k.split(ids, y=None):
        idx_list.append(idx)

    train_idx, val_idx = idx_list[fold_idx]

    train_graph_list = [ids[i] for i in train_idx]
    val_graph_list = [ids[i] for i in val_idx]

    return train_graph_list, val_graph_list

def get_mudata_dataset(args, is_train, preprocess_fn=None):

    #FIXME Add param to param.py
    tokenizer = None
    # args.dataset = 'her2st'
    # args.gene_range = 'all'
    # args.gene_set_list = None
    # args.tissue_data = 'her2st_cls_train.txt'
    image_size = 224

    normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    preprocess_fn = Compose([Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                _convert_to_rgb,
                ToTensor(),
                normalize
            ])

    root = os.path.join('../datasets',args.dataset)
    tissue_file = os.path.join(root, 'data', args.tissue_data) # if is_train else args.val_data 
    tissue_ids = read_file(tissue_file)
    # input_ids = tissue_ids
    train_ids, val_ids = separate_data(tissue_ids, seed=args.seed, n_folds=8, fold_idx=args.fold_idx)
    input_ids = train_ids if is_train else val_ids

    if args.dataset == 'her2st':

        dataset = Her2stDataSet(
            root,
            input_ids,
            transform = preprocess_fn,
            gene_range = args.gene_range,
            gene_set_list = args.gene_set_list,
            include_label=True)

    num_samples = len(dataset)
    shuffle = is_train

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader

if __name__ == "__main__":
    args = lambda: None
    args.fold_idx = 0
    args.seed = 1078
    args.batch_size = 8
    args.workers = 4

    dl = get_mudata_dataset(args, is_train=True)
    print(len(dl))
    # for i, batch in enumerate(dl):
    #     print(i, batch[2])