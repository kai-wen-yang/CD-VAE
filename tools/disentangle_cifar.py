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

from networks.vae import *
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC


def reconst_images(epoch=2, batch_size=64, batch_num=2, dataloader=None, model=None):
    cifar10_dataloader = dataloader

    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(cifar10_dataloader):
            if batch_idx >= batch_num:
                break
            else:
                X, y = X.cuda(), y.cuda().view(-1, )
                _,_, gx, _, _ = model(X)

                grid_X = torchvision.utils.make_grid(X[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_X.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_X)]}, commit=False)
                grid_GX = torchvision.utils.make_grid(gx[:batch_size].data, nrow=8, padding=2, normalize=True)
                wandb.log({"_Batch_{batch}_GX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_GX)]}, commit=False)
                grid_RX = torchvision.utils.make_grid((X[:batch_size] - gx[:batch_size]).data, nrow=8, padding=2,
                                                        normalize=True)
                wandb.log({"_Batch_{batch}_RX.jpg".format(batch=batch_idx): [
                    wandb.Image(grid_RX)]}, commit=False)
    print('reconstruction complete!')


def test(epoch, model, testloader):
    # set model as testing mode
    model.eval()
    # all_l, all_s, all_y, all_z, all_mu, all_logvar = [], [], [], [], [], []
    acc_avg = AverageMeter()
    sparse_avg = AverageMeter()
    top1 = AverageMeter()
    TC = AverageMeter()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            # distribute data to device
            x, y = x.cuda(), y.cuda().view(-1, )
            bs = x.size(0)
            norm = torch.norm(torch.abs(x.view(100, -1)), p=2, dim=1)
            out, hi, gx, mu, logvar = model(x)
            acc_gx = 1 - F.mse_loss(torch.div(gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100
            acc_xd = 1 - F.mse_loss(torch.div(x - gx, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    torch.div(x, norm.unsqueeze(1).unsqueeze(2).unsqueeze(3)), \
                                    reduction='sum') / 100

            acc_avg.update(acc_gx.data.item(), bs)
            # measure accuracy and record loss
            sparse_avg.update(acc_xd.data.item(), bs)
            # measure accuracy and record loss
            prec1, _, _, _ = accuracy(out.data, y.data, topk=(1, 5))
            top1.update(prec1.item(), bs)

            tc = total_correlation(hi, mu, logvar) / bs / args.dim
            TC.update(tc.item(), bs)

        wandb.log({'acc_avg': acc_avg.avg, \
                   'sparse_avg': sparse_avg.avg, \
                   'test-RX-acc': top1.avg, \
                   'test-TC': TC.avg}, commit=False)
        # plot progress
        print("\n| Validation Epoch #%d\t\tRec Acc: %.4f Class Acc: %.4f TC: %.4f" % (epoch, acc_avg.avg, top1.avg, TC.avg))
        reconst_images(epoch=epoch, batch_size=64, batch_num=2, dataloader=testloader, model=model)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        print("Epoch {} model saved!".format(epoch + 1))


def train(args, epoch, model, optimizer, trainloader):
    model.train()
    model.training = True

    loss_avg = AverageMeter()
    loss_rec = AverageMeter()
    loss_ce = AverageMeter()
    loss_entropy = AverageMeter()
    loss_kl = AverageMeter()
    top1 = AverageMeter()

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y, y_b, lam, mixup_index = mixup_data(x, y, alpha=args.alpha)
        x, y, y_b = x.cuda(), y.cuda().view(-1, ), y_b.cuda().view(-1, )
        x, y = Variable(x), [Variable(y), Variable(y_b)]
        bs = x.size(0)
        optimizer.zero_grad()

        out, _, xi, mu, logvar = model(x)

        if args.curriculum:
            if epoch < 100:
                re = 10*args.re
            elif epoch < 200:
                re = 5*args.re
            else:
                re = args.re
        else:
            re = args.re

        l1 = F.mse_loss(xi, x)
        cross_entropy = lam * F.cross_entropy(out, y[0]) + (1. - lam) * F.cross_entropy(out, y[1])
        l2 = cross_entropy
        l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l3 /= bs * 3 * args.dim
        loss = re * l1 + args.ce * l2 + args.kl * l3
        loss.backward()
        optimizer.step()

        prec1, prec5, correct, pred = accuracy(out.data, y[0].data, topk=(1, 5))
        loss_avg.update(loss.data.item(), bs)
        loss_rec.update(l1.data.item(), bs)
        loss_ce.update(cross_entropy.data.item(), bs)
        loss_kl.update(l3.data.item(), bs)
        top1.update(prec1.item(), bs)

        n_iter = (epoch - 1) * len(trainloader) + batch_idx
        wandb.log({'loss': loss_avg.avg, \
                   'loss_rec': loss_rec.avg, \
                   'loss_ce': loss_ce.avg, \
                   'loss_kl': loss_kl.avg, \
                   'acc': top1.avg,
                   're_weight': re,
                   'lr':optimizer.param_groups[0]['lr']}, step=n_iter)
        if (batch_idx + 1) % 30 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(
                '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_entropy: %.4f Loss_kl: %.4f Acc@1: %.3f%%'
                % (epoch, args.epochs, batch_idx + 1,
                   len(trainloader), loss_avg.avg, loss_rec.avg, loss_ce.avg, loss_entropy.avg, loss_kl.avg, top1.avg))


def main(args):
    learning_rate = 1.e-3
    learning_rate_min = 2.e-4
    CNN_embed_dim = args.dim
    feature_dim = args.fdim
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    if (args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Model
    print('\n[Phase 2] : Model setup')
    model = CVAE_cifar(d=feature_dim, z=CNN_embed_dim)

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    optimizer = AdamW([
        {'params': model.parameters()}
    ], lr=learning_rate, betas=(0.9, 0.999), weight_decay=1.e-6)

    if args.optim == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50,
                                                        eta_min=learning_rate_min)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=0.1, last_epoch=-1)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))

    start_epoch = 1
    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        train(args, epoch, model, optimizer, trainloader)
        scheduler.step()
        if epoch % 10 == 0:
            test(epoch, model, testloader)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time

    wandb.finish()
    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5/', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--optim', default='cosine', type=str, help='optimizer')
    parser.add_argument('--alpha', default=2.0, type=float, help='mix up')
    parser.add_argument('--epochs', default=300, type=int, help='training_epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--dim', default=2048, type=int, help='CNN_embed_dim')
    parser.add_argument('--T', default=50, type=int, help='Cosine T')
    parser.add_argument('--fdim', default=32, type=int, help='featdim')
    parser.add_argument('--step', nargs='+', type=int)
    parser.add_argument('--re', default=1.0, type=float, help='reconstruction weight')
    parser.add_argument('--curriculum', default=True,
                        help='Curriculum for reconstruction term which helps for better convergence')
    parser.add_argument('--kl', default=0.2, type=float, help='kl weight')
    parser.add_argument('--ce', default=0.2, type=float, help='cross entropy weight')
    args = parser.parse_args()
    wandb.init(config=args, name=args.save_dir.replace("results/", ''))
    set_random_seed(args.seed)
    main(args)
