CUDA_VISIBLE_DEVICES=0 python tools/disentangle_cifar.py --save_dir results/disentangle_cifar_ce0.2 --dim 2048 --fdim 32 --ce 0.2 --optim cosine



#CUDA_VISIBLE_DEVICES=0 python tools/disentangle_cifar.py --save_dir results/disentangle_cifar_ce1 --dim 2048 --fdim 32 --ce 0.1 --optim consine
#
#CUDA_VISIBLE_DEVICES=0 python tools/disentangle_cifar.py --save_dir results/disentangle_cifar_ce0.2_curriculum_entropy --dim 2048 --fdim 32 --ce 0.2 --re 1 --kl 0.2 --optim consine --curriculum
#
