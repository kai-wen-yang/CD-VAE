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

from networks.adv_vae import *
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC
from utils.normalize import *
from advex.attacks import *
from evaluation import evaluate_cdvae
normalize = CIFARNORMALIZE(32)


def Incorrect_Logits(logits, labels, margin):
    max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
    top_max, second_max = max_2_logits.chunk(2, dim=1)
    top_argmax, second_argmax = argmax_2_logits.chunk(2, dim=1)
    labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
    labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
    max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
    correct_logits = torch.gather(logits, 1, labels.view(-1, 1))
    return ((correct_logits - max_incorrect_logits)<margin).view(-1)


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = args.lr * (0.1 ** factor)

        """Warmup"""
        if epoch < 5:
            lr = lr * float(1 + step) / (5. * len_epoch)

        # if(args.local_rank == 0):
        #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--save_dir', default='./results/autoaug_new_8_0.5/', type=str, help='save_dir')
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--batch_size', default=128, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--epochs', default=150, type=int, help='training_epochs')
    parser.add_argument('--dim', default=2048, type=int, help='CNN_embed_dim')
    parser.add_argument('--fdim', default=32, type=int, help='featdim')
    parser.add_argument('--margin', default=8.0, type=float, help='margin')
    parser.add_argument('--re', default=1.0, type=float, help='re weight')
    parser.add_argument('--kl', default=0.01, type=float, help='kl weight')
    parser.add_argument('--cr', default=1.0, type=float, help='cross entropy weight')
    parser.add_argument('--cg', default=1.0, type=float, help='cross entropy weight')
    parser.add_argument("--model_path", type=str, default="./pretrained/wide_resnet.pth")
    parser.add_argument("--vae_path", type=str, default="./pretrained/cdvae2.pth")
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='clip gradients to this value')
    args = parser.parse_args()

    wandb.init(config=args, name=args.save_dir.replace("results/", ''))
    set_random_seed(args.seed)
    CNN_embed_dim = args.dim
    feature_dim = args.fdim
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()

    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Model
    print('\n[Phase 2] : Model setup')
    vae = CVAE_cifar(d=feature_dim, z=CNN_embed_dim, with_classifier=False)
    model_r = Wide_ResNet(28, 10, 0.3, 10)
    model_g = Wide_ResNet(28, 10, 0.3, 10)
    if use_cuda:
        vae.cuda()
        vae = torch.nn.DataParallel(vae, device_ids=range(torch.cuda.device_count()))

        model_r.cuda()
        model_r = torch.nn.DataParallel(model_r, device_ids=range(torch.cuda.device_count()))

        model_g.cuda()
        model_g = torch.nn.DataParallel(model_g, device_ids=range(torch.cuda.device_count()))

        cudnn.benchmark = True

    save_model = torch.load(args.vae_path)
    model_dict = vae.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    vae.load_state_dict(model_dict)

    model_g.load_state_dict(torch.load(args.model_path))

    model_dict = model_r.state_dict()
    state_dict = {k.replace('classifier.', ''): v for k, v in save_model.items() if
                  k.replace('classifier.', '') in model_dict.keys()}
    model_dict.update(state_dict)
    model_r.load_state_dict(model_dict)

    optimizer = optim.SGD([
        {'params': vae.parameters()},
        {'params': model_r.parameters()},
        {'params': model_g.parameters()}],
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=2e-4)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.epochs))

    iteration = 0
    attack = AttackV2(model_g, vae, num_iterations=10, loss='margin')
    validation_attacks = [NoAttack(),
                          AttackV2(model_g, vae, num_iterations=100),
                          AttackV2(model_g, vae, num_iterations=100, norm='l2', eps_max=1.0),
                          AttackV2(model_g, vae, num_iterations=100, loss='margin'),
                          AttackV2(model_g, vae, num_iterations=100, norm='l2', eps_max=1.0,loss='margin')]
    elapsed_time = 0

    def run_iter(inputs, labels, iteration):
        model_r.eval()  # set model to eval to generate adversarial examples
        model_g.eval()
        vae.eval()

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        bs = inputs.size(0)
        with torch.no_grad():
            gx, _, _ = vae(normalize(inputs))
            orig_logits = model_g(gx)
            orig_accuracy, _, _, _ = accuracy(orig_logits.data, labels.data, topk=(1, 5))
            to_attack = orig_logits.argmax(1) == labels

        adv_inputs = inputs.clone()
        if to_attack.sum()>0:
            adv_inputs[to_attack]= attack(inputs[to_attack], labels[to_attack])

        with torch.no_grad():
            gx, _, _ = vae(normalize(adv_inputs))
            adv_logits_g = model_g(gx)
            adv_accuracy, _, _, _ = accuracy(adv_logits_g.data, labels.data, topk=(1, 5))
            incorrect = Incorrect_Logits(adv_logits_g, labels, args.margin)

        optimizer.zero_grad()
        model_r.train()
        model_g.train()
        vae.train()

        gx, mu, logvar = vae(normalize(adv_inputs))
        logits_g = model_g(gx)
        logits_r = model_r(normalize(adv_inputs)-gx)

        l1 = F.mse_loss(gx, normalize(inputs))

        l2 = args.cr * F.cross_entropy(logits_r, adv_logits_g.argmax(1)) \
           + args.cg * F.cross_entropy(logits_g[incorrect], labels[incorrect])

        l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l3 /= bs * 3 * args.dim

        loss = args.re * l1 +  l2 + args.kl * l3
        loss.backward()
        nn.utils.clip_grad_value_(model_r.parameters(), args.clip_grad)
        nn.utils.clip_grad_value_(model_g.parameters(), args.clip_grad)
        nn.utils.clip_grad_value_(vae.parameters(), args.clip_grad)
        optimizer.step()

        wandb.log({'loss': loss.item()}, step=iteration)
        wandb.log({'loss1': l1.item()}, step=iteration)
        wandb.log({'loss2': l2.item()}, step=iteration)
        wandb.log({'loss3': l3.item()}, step=iteration)
        wandb.log({'adversarial_accuracy': adv_accuracy.item()}, step=iteration)
        wandb.log({'orig_accuracy': orig_accuracy.item()}, step=iteration)
        wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=iteration)

        print(f'ITER {iteration:06d}',
              f'loss: {loss.item():.2f}',
              f'loss1: {l1.item():.2f}',
              f'loss2: {l2.item():.2f}',
              f'loss3: {l3.item():.2f}',
              f'orig_accuracy: {orig_accuracy.item():5.1f}%',
              f'acc_adv: {adv_accuracy.item() :5.1f}%',
              sep='\t')

    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + args.epochs):
        start_time = time.time()
        for batch_index, (inputs, labels) in enumerate(trainloader):
            adjust_learning_rate(optimizer, epoch, iteration, len(trainloader))
            run_iter(inputs, labels, iteration)
            iteration += 1

        if epoch % 10 == 1:
            print("\n| Validation Epoch #%d\t\t" % (epoch))
            evaluate_cdvae.evaluate(wandb, model_r, model_g, vae, testloader, validation_attacks, 10)
            torch.save(model_r.state_dict(),
                       os.path.join(args.save_dir, 'model_r_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
            torch.save(model_g.state_dict(),
                       os.path.join(args.save_dir, 'robust_model_g_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
            torch.save(vae.state_dict(),
                       os.path.join(args.save_dir, 'robust_vae_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
            print("Epoch {} model saved!".format(epoch + 1))

    evaluate_cdvae.evaluate(wandb, model_r, model_g, vae, testloader, validation_attacks)
    torch.save(model_r.state_dict(),
               os.path.join(args.save_dir, 'model_r_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(model_g.state_dict(),
               os.path.join(args.save_dir, 'robust_model_g_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(vae.state_dict(),
               os.path.join(args.save_dir, 'robust_vae_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    print("Epoch {} model saved!".format(epoch + 1))
    wandb.finish()
