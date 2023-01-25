import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

import wandb

from util import read_file, get_scheduler, EarlyStopper
from evaluate import Evaluator, plot_confusion_matrix

from dataloader import separate_data, _convert_to_rgb
from her2st_data import Her2stDataSet 

from swin_transformer import swin_tiny_patch4_window7_224, ConvStem

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer, scheduler, train_evaluator):
    model.train()

    loss_accum = 0
    y_true = []
    y_pred = []
    pred = None
    for step, batch in enumerate(tqdm(loader)):

        images, _, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        optimizer.zero_grad()

        loss = multicls_criterion(pred.to(torch.float32), labels.view(-1,))

        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().numpy()

        y_true.append(labels.view(-1,1).detach().cpu())
        y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    scheduler.step()

    avg_loss = loss_accum / len(loader)
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return train_evaluator.eval(input_dict), avg_loss

def eval(model, device, loader, evaluator):
    model.eval()

    loss_accum = 0
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        batch = batch.to(device)
        images, _, labels = batch

        with torch.no_grad():
            pred = model(images)

            loss = multicls_criterion(pred.to(torch.float32), labels.view(-1,))
            loss_accum += loss.detach().cpu().numpy()

        y_true.append(labels.view(-1,1).detach().cpu())
        y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    avg_loss = loss_accum / len(loader)

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), avg_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--image_encoder', type=str, default='ctranspath',
                        help='ctranspath | vit-path | kimia')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_policy', type=str, default='cosine',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--l2_weight_decay', type=float, default=0.001,
                        help='The weight decay for L2 Norm in Adam optimizer')

    parser.add_argument('--dataset', type=str, default="her2st",
                        help='dataset name (default: her2st)')
    parser.add_argument('--phase', type=str, default="train",
                        help='dataset phase : train | test | plot')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='Number of classes')
    
    parser.add_argument('--n_folds', type=int, default=5,
                        help='total number of folds.')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--config_file', type=str, default="experiments/configs/config.yaml",
                        help='parameter and hyperparameter config i.e all values for model and dataset parameters')

    args = parser.parse_args()

    wandb.init(project="CLIP", config=args.config_file)
    wandb.run.name = wandb.run.name + "_fold_" + str(args.fold_idx)
    wandb.config.update({'fold_idx': args.fold_idx,
                         'run_name': wandb.run.name,
                         'log_path': os.path.join('logs', wandb.run.name),
                         'device': args.device}, allow_val_change=True)

    config = wandb.config
    os.makedirs(config.log_path, exist_ok=True)

    print(config)

    ### set up seeds and gpu device
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting

    if config.dataset == 'her2st':
        config.gene_range = 'all'
        config.gene_set_list = None
        config.image_size = 224
        dataset_class = Her2stDataSet

        normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        preprocess_fn = Compose([Resize(config.image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(config.image_size),
                _convert_to_rgb,
                ToTensor(),
                normalize])

    root = os.path.join('../datasets', config.dataset)

    wsi_file = os.path.join('../datasets', config.dataset, 'data', '%s_%s.txt' % (config.dataset, config.phase))
    wsi_ids = read_file(wsi_file)

    train_wsi_ids, valid_wsi_ids = separate_data(wsi_ids, config.seed, config.n_folds, config.fold_idx)

    isTrain = True if config.phase == 'train' else False

    train_dataset = dataset_class(root, 
                                  train_wsi_ids, 
                                  transform=preprocess_fn, 
                                  gene_range=config.gene_range, 
                                  gene_set_list=config.gene_set_list,
                                  include_label=True)

    valid_dataset = dataset_class(root, 
                                  valid_wsi_ids,
                                  transform=preprocess_fn,
                                  gene_range=config.gene_range,
                                  gene_set_list=config.gene_set_list,
                                  include_label=True)

    train_loader = DataLoader(train_dataset, 
                            batch_size=config.batch_size, 
                            shuffle=isTrain,
                            num_workers=config.num_workers,
                            pin_memory=True,
                            sampler=None,
                            drop_last=isTrain)

    valid_loader = DataLoader(valid_dataset, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=config.num_workers,
                            pin_memory=True,
                            sampler=None)

    train_evaluator = Evaluator(train_dataset)
    evaluator = Evaluator(valid_dataset) 

    if config.image_encoder == 'ctranspath':
        model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False).to(device)
        
        # load Pre-Trained Model
        td = torch.load(r'./experiments/ctranspath.pth')
        model.load_state_dict(td['model'], strict=False)
        linear_keyword = 'head'

        for name, param in model.named_parameters():
            if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
                param.requires_grad = False
        # init the fc layer
        getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
        getattr(model, linear_keyword).bias.data.zero_()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # weight, bias
    else:
        raise ValueError('Invalid Encoder type')

    # print()

    optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.l2_weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.1)

    config.iterations_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, config)

    valid_curve = []
    train_curve = []
    valid_cm_plots = []

    wandb.watch(model)

    print('Training...')
    early_stopper = EarlyStopper(patience=10, min_delta=0.001, after_epoch=120)

    for epoch in range(config.n_epochs+1):
        print("=====Epoch {}".format(epoch))
        print("Train Loader length", len(train_loader))
        
        train_perf, train_loss = train(model, device, train_loader, optimizer, scheduler, train_evaluator) # Avg loss per epoch

        print('Evaluating...')
        valid_perf, valid_loss = eval(model, device, valid_loader, evaluator)

        # print('Train', train_perf)
        print('Validation', valid_perf)

        metrics = {'train/loss': train_loss,
                   'train/acc': train_perf['acc'],
                   'valid/loss': valid_loss,                   
                   'valid/acc': valid_perf['acc']
                  }

        wandb.log(metrics)

        if early_stopper.early_stop(valid_loss, epoch):             
            break
    
    print('Finished Training')
    print('Final validation score: {}'.format(valid_perf))

    final_cm_plot = plot_confusion_matrix(valid_perf['cm'], list(valid_dataset.classdict.keys()), title='fold{}(Val Accuracy={:0.2f})'.format(config.fold_idx+1, valid_perf['acc']))
    wandb.log({"ConfusionMatrix": final_cm_plot})

    with open(os.path.join(config.log_path, config.run_name+'valid_perf.txt'), 'w') as f:
        for key, value in valid_perf.items():
            f.write('%s:%s\n' % (key, value))

    save_path = os.path.join(config.log_path, config.run_name+'.pth')
    torch.save(model.cpu().state_dict(), save_path)
    
    wandb.finish()

if __name__ == "__main__":
    main()