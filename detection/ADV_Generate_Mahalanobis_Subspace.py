from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
import models
import os
import lib_generation
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
import pdb

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='../../data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', required=True, help='FGSM | BIM | PGD | CW')
parser.add_argument('--vae_path', default='./data/96.32/model_epoch252.pth', help='folder to output results')
args = parser.parse_args()
print(args)

def main():
    # set the path to pre-trained model and output
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])
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
    train_loader, _ = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    test_clean_data = torch.load(args.outf + 'clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_adv_data = torch.load(args.outf + 'adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_noisy_data = torch.load(args.outf + 'noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))
    test_label = torch.load(args.outf + 'label_%s_%s_%s.pth' % (args.net_type, args.dataset, args.adv_type))

    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.module.feature_list(temp_x-vae(temp_x))[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, vae, args.num_classes, feature_list, train_loader)

    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('\nNoise: ' + str(magnitude))
        for i in range(num_output):
            M_in \
            = lib_generation.get_Mahalanobis_score_adv(model, vae, test_clean_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_out \
            = lib_generation.get_Mahalanobis_score_adv(model, vae, test_adv_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_noisy \
            = lib_generation.get_Mahalanobis_score_adv(model, vae, test_noisy_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_noisy = np.asarray(M_noisy, dtype=np.float32)
            if i == 0:
                Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
            else:
                Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
        Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))

        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_pos)
        file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.dataset, args.adv_type))

        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)



if __name__ == '__main__':
    main()

