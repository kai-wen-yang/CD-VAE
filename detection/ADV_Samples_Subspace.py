"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import data_loader
import numpy as np
import models
import os
import lib.adversary as adversary
from lib.attacks import DeltaAttack
import pdb
from torchvision import transforms
from torch.autograd import Variable
import lib_generation
parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', default='../../data/', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | BIM | PGD | CW')
parser.add_argument('--vae_path', default='./data/96.32/model_epoch252.pth', help='folder to output results')
parser.add_argument('--pertubation', type=float, default=8/255, help='adversarial pertubation')
parser.add_argument('--steps', type=int, default=5, help='adversarial iteration')
args = parser.parse_args()
print(args)


def main():
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.makedirs(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    if args.adv_type == 'FGSM':
        adv_noise = args.pertubation

    in_transform = transforms.Compose([transforms.ToTensor(), \
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])

    min_pixel = -1.9894736842105263
    max_pixel = 2.126488706365503
    if args.adv_type == 'FGSM':
        random_noise_size = 0.25 / 4
    else:
        random_noise_size = 0.13 / 2
    model = models.Wide_ResNet(28, 10, 0.3, 10)
    model = nn.DataParallel(model)
    model_dict = model.state_dict()
    save_model = torch.load(args.vae_path)
    state_dict = {k.replace('classifier.',''): v for k, v in save_model.items() if k.replace('classifier.','') in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model.eval()
    print('load model: ' + args.net_type)

    vae = models.CVAE(d=32, z=2048)
    vae = nn.DataParallel(vae)
    model_dict = vae.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    vae.load_state_dict(model_dict)
    vae.cuda()
    vae.eval()

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    
    print('Attack: ' + args.adv_type  +  ', Dist: ' + args.dataset + '\n')
    model.eval()
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0
    
    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()
    print('Generating testset:')
    selected_list = []
    selected_index = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model( data - vae(data))

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()

        noisy_data = torch.add(data.data,  torch.randn(data.size()).cuda(), alpha=random_noise_size)
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
            
        # generate adversarial
        model.zero_grad()
        vae.zero_grad()

        if args.adv_type == 'FGSM':
            inputs = Variable(data.data, requires_grad=True)
            output = model(inputs - vae(inputs))
            loss = criterion(output, target)
            loss.backward()
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float()-0.5)*2
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2470))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.2435))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2616))
            adv_data = torch.add(inputs.data, gradient, alpha=adv_noise)

            adv_data = torch.clamp(adv_data, min_pixel, max_pixel)

        elif args.adv_type == 'BIM': 
            attack = DeltaAttack(model, vae, num_iterations=5, datasets=args.dataset, rand_init=False)
            adv_data = attack(data, target)

        elif args.adv_type == 'PGD':
            attack = DeltaAttack(model, vae, num_iterations=5, datasets=args.dataset)
            adv_data = attack(data, target)

        elif args.adv_type == 'CW':
            attack = DeltaAttack(model, vae, num_iterations=5, datasets=args.dataset, loss='margin')
            adv_data = attack(data, target)

        elif args.adv_type == 'PGD-L2':
            attack = DeltaAttack(model, vae, eps_max=1.0, num_iterations=5, datasets=args.dataset, norm='l2')
            adv_data = attack(data, target)
        # measure the noise 
        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)

        if total == 0:
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)
        with torch.no_grad():
            output = model(Variable(adv_data)-vae(Variable(adv_data)))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_adv = pred.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
        with torch.no_grad():
            output = model(Variable(noisy_data)-vae(Variable(noisy_data)))
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()
        
        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
            
        total += data.size(0)

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))

    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))


if __name__ == '__main__':
    main()
