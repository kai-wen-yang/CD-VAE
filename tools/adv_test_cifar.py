
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
from perceptual_advex.attacks import StAdvAttack
import wandb
import os
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys
import wandb
sys.path.append('.')
from typing import Dict, List
from networks.adv_vae import *
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC
from utils.normalize import *
from advex.attacks import *
from perceptual_advex.attacks import *

normalize = CIFARNORMALIZE(32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')
    parser.add_argument('--dim', default=2048, type=int, help='CNN_embed_dim')
    parser.add_argument('--fdim', default=32, type=int, help='featdim')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    parser.add_argument("--model_path", type=str, default="./results/v3cr1.0_cg1.0_kl0.1/model_g_epoch42.pth")
    parser.add_argument("--vae_path", type=str, default="./results/v3cr1.0_cg1.0_kl0.1/vae_epoch42.pth")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cd_vae = CD_VAE(args.vae_path, args.model_path)
    wandb.init(config=args)
    if use_cuda:
        cd_vae.cuda()
        cudnn.benchmark = True

    cd_vae.eval()

    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]
    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    for batch_index, (inputs, labels) in enumerate(testloader):
        print(f'BATCH {batch_index:05d}')

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            adv_inputs = attack(inputs, labels)
            with torch.no_grad():
                adv_logits = cd_vae(adv_inputs)
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  sep='\t')
            batches_correct[attack_name].append(batch_correct)

    print('OVERALL')
    accuracies = []
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              sep='\t')
        accuracies.append(accuracy)
