import torch
from torch import nn
from torch.nn import functional as F
import pdb
import lib.runutils
from torch.autograd import Variable
import operator as op

from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
def _var2numpy(var):
    """
    Make Variable to numpy array. No transposition will be made.

    :param var: Variable instance on whatever device
    :type var: Variable
    :return: the corresponding numpy array
    :rtype: np.ndarray
    """
    return var.data.cpu().numpy()

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

def get_eps_params(base_eps, resol):
    eps_list = []
    max_list = []
    min_list = []
    for i in range(3):
        eps_list.append(torch.full((resol, resol), base_eps, device='cuda'))
        min_list.append(torch.full((resol, resol), 0., device='cuda'))
        max_list.append(torch.full((resol, resol), 255., device='cuda'))

    eps_t = torch.unsqueeze(torch.stack(eps_list), 0)
    max_t = torch.unsqueeze(torch.stack(max_list), 0)
    min_t = torch.unsqueeze(torch.stack(min_list), 0)
    return eps_t, max_t, min_t

def get_cifar_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), CIFAR_STD[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

class CIFARNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class CIFARINNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x


class _MahalanobisLoss(nn.Module):
    def __init__(self):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_MahalanobisLoss, self).__init__()

    def forward(self, feature, mean, inverse_cov):
        mahalaonbis_loss = 0
        zero_f = feature - mean
        mahalaonbis_loss = torch.mm(torch.mm(zero_f, inverse_cov), zero_f.t()).diag()
        mahalaonbis_loss = torch.mean(mahalaonbis_loss)
        return mahalaonbis_loss


class _MahalanobisEnsembleLoss(nn.Module):
    def __init__(self):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_MahalanobisEnsembleLoss, self).__init__()

    def forward(self, feature, mean, inverse_cov, weight, top2_index):
        mahalaonbis_loss = 0
        for i in range(len(mean)):
            temp_loss = 0
            final_mean = mean[i].index_select(0, top2_index.cuda())
            final_mean = Variable(final_mean)
            zero_f = feature[i] - final_mean
            temp_loss = torch.mm(torch.mm(zero_f, Variable(inverse_cov[i])), zero_f.t()).diag()
            mahalaonbis_loss += weight[i]*torch.mean(temp_loss)
        return mahalaonbis_loss


class MarginLoss(nn.Module):
    """
    Calculates the margin loss max(kappa, (max z_k (x) k != y) - z_y(x)),
    also known as the f6 loss used by the Carlini & Wagner attack.
    """

    def __init__(self, kappa=float('inf'), targeted=False):
        super().__init__()
        self.kappa = kappa
        self.targeted = targeted

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1))

        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, second_argmax = argmax_2_logits.chunk(2, dim=1)
        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
        max_incorrect_index = labels_eq_max * second_argmax + labels_ne_max * top_argmax
        if self.targeted:
            return (correct_logits - max_incorrect_logits) \
                .clamp(max=self.kappa).squeeze()
        else:
            return (max_incorrect_logits - correct_logits) \
                .clamp(max=self.kappa).squeeze()


class DeltaAttack(nn.Module):
    def __init__(self, model,  vae, eps_max=8/255, step_size=None,  num_iterations=7, datasets = 'cifar10', norm='linf', rand_init=True, scale_each=False, loss='ce'):
        super().__init__()
        self.nb_its = num_iterations
        self.eps_max = eps_max
        if step_size is None:
            step_size = eps_max / (self.nb_its ** 0.5)
        self.step_size = step_size

        self.norm = norm
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.loss = loss

        if self.loss == 'margin':
            self.criterion = MarginLoss(kappa=1000)
        else:
            self.criterion = nn.CrossEntropyLoss().cuda()
        self.model = model
        self.vae = vae
        self.datasets = datasets
        if self.datasets == 'cifar10':
            self.normalize = CIFARNORMALIZE(32)
            self.innormalize = CIFARINNORMALIZE(32)

    def _init(self, shape, eps):
        if self.rand_init:
            if self.norm == 'linf':
                init = torch.rand(shape, dtype=torch.float32, device='cuda') * 2 - 1
            elif self.norm == 'l2':
                init = torch.randn(shape, dtype=torch.float32, device='cuda')
                init_norm = torch.norm(init.view(init.size()[0], -1), 2.0, dim=1)
                normalized_init = init / init_norm[:, None, None, None]
                dim = init.size()[1] * init.size()[2] * init.size()[3]
                rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/dim)
                init = normalized_init * rand_norms[:, None, None, None]
            else:
                raise NotImplementedError
            init = eps[:, None, None, None] * init
            init.requires_grad_()
            return init
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')

    def forward(self, img, labels):
        img = self.innormalize(img) #0-1
        base_eps = self.eps_max * torch.ones(img.size()[0], device='cuda')
        step_size = self.step_size * torch.ones(img.size()[0], device='cuda')

        img = img.detach()
        img.requires_grad = True
        delta = self._init(img.size(), base_eps)

        s = self.model(self.normalize(img + delta)
            - self.vae(self.normalize(img + delta)))
        if self.norm == 'l2':
            l2_max = base_eps
        for it in range(self.nb_its):
            loss = self.criterion(s, labels)

            if self.loss == 'margin':
                loss.sum().backward()
            else:
                loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            grad = delta.grad.data

            if self.norm == 'linf':
                grad_sign = grad.sign()
                delta.data = delta.data + step_size[:, None, None, None] * grad_sign
                delta.data = torch.max(torch.min(delta.data, base_eps[:, None, None, None]), -base_eps[:, None, None, None])
                delta.data = torch.clamp(img.data + delta.data, 0., 1.) - img.data
            elif self.norm == 'l2':
                batch_size = delta.data.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None, None, None]
                delta.data = delta.data + step_size[:, None, None, None]   * normalized_grad
                l2_delta = torch.norm(delta.data.view(batch_size, -1), 2.0, dim=1)
                # Check for numerical instability
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), l2_max / l2_delta)
                delta.data *= proj_scale[:, None, None, None]
                delta.data = torch.clamp(img.data + delta.data, 0., 1.) - img.data
            else:
                raise NotImplementedError

            if it != self.nb_its - 1:
                s = self.model(self.normalize(img + delta)
                               - self.vae(self.normalize(img + delta)))
                delta.grad.data.zero_()
        delta.data[torch.isnan(delta.data)] = 0
        adv_sample = img + delta
        adv_sample = torch.clamp(adv_sample.detach(), 0, 1)
        return self.normalize(adv_sample)
