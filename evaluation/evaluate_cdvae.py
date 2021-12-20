from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys

sys.path.append('.')

from networks import *
from utils.set import *
from utils.normalize import *
from advex.attacks import *
normalize = CIFARNORMALIZE(32)

def run_iter(wandb, batch_idx, x, y, model_r, model_g, vae, attack):
    attack_name = attack.__class__.__name__
    x, y = x.cuda(), y.cuda().view(-1, )
    adv_x = attack(x, y)
    rescont_bs = 64

    gx, _, _ = vae(normalize(adv_x))
    logits_r = model_r(normalize(adv_x) - gx)
    logits_g = model_g(gx)

    prec1_g, _, _, _ = accuracy(logits_g.data, y.data, topk=(1, 5))
    prec1_r, _, _, _ = accuracy(logits_r.data, y.data, topk=(1, 5))

    if batch_idx <= 1:
        grid_X = torchvision.utils.make_grid(adv_x[:rescont_bs].data, nrow=8, padding=2, normalize=True)
        wandb.log({"_{attack}/_{batch}_X.jpg".format(batch=batch_idx, attack=attack_name): [
            wandb.Image(grid_X)]}, commit=False)
        grid_Xi = torchvision.utils.make_grid(gx[:rescont_bs].data, nrow=8, padding=2, normalize=True)
        wandb.log({"_{attack}/_{batch}_GX.jpg".format(batch=batch_idx, attack=attack_name): [
            wandb.Image(grid_Xi)]}, commit=False)
        grid_X_Xi = torchvision.utils.make_grid((normalize(adv_x)[:rescont_bs] - gx[:rescont_bs]).data, nrow=8,
                                                padding=2,
                                                normalize=True)
        wandb.log({"_{attack}/_{batch}_RX.jpg".format(batch=batch_idx, attack=attack_name): [
            wandb.Image(grid_X_Xi)]}, commit=False)
    return prec1_r, prec1_g

def test(wandb, model_r, model_g, vae, testloader, attack, val_num=None):
    attack_name = attack.__class__.__name__
    model_r.eval()
    model_g.eval()
    vae.eval()

    top1_g = AverageMeter()
    top1_r = AverageMeter()

    for batch_idx, (x , y) in enumerate(testloader):
        bs = x.size(0)
        if val_num:
            if batch_idx >= val_num:
                break
        prec1_r, prec1_g = run_iter(wandb, batch_idx, x, y, model_r, model_g, vae, attack)
        top1_g.update(prec1_g.item(), bs)
        top1_r.update(prec1_r.item(), bs)

    wandb.log({f'{attack_name}/test-XG-acc': top1_g.avg, \
               f'{attack_name}/test-XR-acc': top1_r.avg}, commit=False)
    # plot progress
    print('Attack:{}'.format(attack_name))
    print("| XG: %.2f%% XR: %.2f%%" % (top1_g.avg, top1_r.avg))

def evaluate(wandb, model_r, model_g, vae, testloader, validation_attacks, val_num=None):
    for val_attack in validation_attacks:
        test(wandb, model_r, model_g, vae, testloader, val_attack, val_num)
