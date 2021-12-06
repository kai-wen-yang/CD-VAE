# CD-VAE

Official implementation:
- Class-Disentanglement and Applications in Adversarial Detection and Defense, NeurIPS 2021. ([Paper](https://openreview.net/pdf?id=jFMzBeLyTc0))

<div align="center">
  <img src="cd_vae.png" width="1000px" />
  <p>CD-VAE</p>
</div>

For any questions, contact (kwyang@mail.ustc.edu.cn).

## Requirements

1. [Python](https://www.python.org/)
2. [Pytorch](https://pytorch.org/)
3. [Wandb](https://wandb.ai/site)
4. [Torchvision](https://pytorch.org/vision/stable/index.html)
5. [Perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex)
6. [Robustness](https://github.com/MadryLab/robustness)

## Pretrained Models
```
cd CD-VAE
mkdir pretrained
```
Downloads pretrained models and put them in folder ./pretrained
1. [cd-vae-1](https://drive.google.com/file/d/1I2yuYQGEYRgqd1oQazq6goDbU2nwUvU_/view?usp=sharing) (gamma=0.2, for adversarial detection)
2. [cd-vae-2](https://drive.google.com/file/d/1I2yuYQGEYRgqd1oQazq6goDbU2nwUvU_/view?usp=sharing) (for initializing adversarial training model)
3. [wide_resnet](https://drive.google.com/file/d/1I2yuYQGEYRgqd1oQazq6goDbU2nwUvU_/view?usp=sharing) (trained on clean data x, for initializing adversarial training model)

## 1. Class-Disentangled VAE
```
cd CD-VAE
python tools/disentangle_cifar.py --save_dir results/disentangle_cifar_ce0.2 --ce 0.2 --optim cosine
```
--ce: Weight of the cross-entropy loss, i.e., gamma in the paper. You can try different values of it (e.g., ce=0.02, 0.2, 2) to control the reconstruction-classification trade-off.
--save_dir: Folder to save the model checkpoint and training log.
--optim: Scheduler of learning rate, we support cosine decay and stage decay now.

## 2. Adversarial Detection

## Citation

If you find this repo useful for your research, please consider citing the paper
```
@article{yang2021class,
  title={Class-Disentanglement and Applications in Adversarial Detection and Defense},
  author={Yang, Kaiwen and Zhou, Tianyi and Tian, Xinmei and Tao, Dacheng and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
