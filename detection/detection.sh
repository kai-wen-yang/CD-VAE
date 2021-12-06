CUDA_VISIBLE_DEVICES=1 python ADV_Samples_Subspace.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Samples_Subspace.py --dataset cifar10 --net_type resnet --adv_type BIM --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Samples_Subspace.py --dataset cifar10 --net_type resnet --adv_type PGD --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Samples_Subspace.py --dataset cifar10 --net_type resnet --adv_type CW --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Samples_Subspace.py --dataset cifar10 --net_type resnet --adv_type PGD-L2 --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;

CUDA_VISIBLE_DEVICES=1 python ADV_Generate_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Generate_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type BIM --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Generate_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type PGD --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Generate_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type CW --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;
CUDA_VISIBLE_DEVICES=1 python ADV_Generate_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type PGD-L2 --gpu 0 --outf ./data/cd-vae-1/ --vae_path ../pretrained/cd-vae-1.pth;

CUDA_VISIBLE_DEVICES=1 python ADV_Regression_Subspace.py --net_type resnet --outf ./data/cd-vae-1/;
