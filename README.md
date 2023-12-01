# Diffusion
Deep Learning Project on Diffusion Models for Image Generation based on [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)'s paper by Karras et al.

## Generated samples

### Unconditional

**CelebA epoch 124, stochastic heun, 33M parameters U-Net model:**

| <img src="src/images/stochastic_heun/v1/uncond_samples_celeba_16_epoch_124.png" width=500> | <img src="src/images/stochastic_heun/v1/iterative_denoising_process_celeba_epoch_124.png" width=500> |
|:--:| :--:|
| *Randomly generated faces* | *An iterative denoising process* |

### Conditional with Classifier-Free Guidance


| <img src="src/images/euler/all_fashionmnist_10.png" width=500> | <img src="src/images/euler/all_cifar10_10_cfgscale_2_5.png" width=500> |
|:--:| :--:|
| <img src="src/images/euler/all_fashionmnist_90.png" width=500> | <img src="src/images/euler/all_cifar10_90_cfgscale_2_5.png" width=500> |
| *FashionMNIST cfg.scale=1, Euler method* | *CIFAR-10 cfg.scale=2.5, Euler method* |

**Cherry-picked generated horse (cfg.scale=2.5):**
<p align="center">
<img src="src/images/euler/horse_cherrypicked_cfgscale_2_5.png" width=500>
</p>

# How To Use?

## Training

If you want to train a model from scratch, please make sure to delete `checkpoints/checkpoint.pth`, otherwise training will resume.

```bash
python src/trainer.py
```


## Sampling

- Unconditional generation

```bash
python src/sampler.py
```

- Classifier-Free Guidance (CFG) for a single class

```bash
python src/sampler.py common.sampling.label=<class-id> common.sampling.cfg_scale=1
```

- Classifier-Free Guidance (CFG) for all classes

```bash
python src/sampler_all.py common.sampling.cfg_scale=1
```


# Tests

- No code coverage report

```
python -m unittest discover -s tests/
```

- If want code coverage:

```
python -m coverage run -m unittest discover -s tests/
```

```
python -m coverage report --omit=*python3*
```

# Notes

## Practical

- Bigger network for more capacity. Going from 2M to 33M parameters: randomly positioned eyes (and more than $2$) to recognizable faces (probably thanks to the Multi-head self-attention layers and bigger receptive fields!)
- Trade-off between the number of parameters and batch size for higher resolution images due to VRAM limits.
- We have to place the Multi-head self-attention layers in lower spatial resolution due to the quadratic complexity in attention. A spatial resolution $32^2=32 \times 32$ or $16^2 = 16 \times 16$ is intuitively enough to capture the long-range contextual information/dependencies and costs way less than a spatial resolution of $128^2$. On the other hand convolution layers, at both high and low spatial resolution, fix "local" inconsistencies (up to the receptive field size).
- TODO: ablation of Multi-head self-attention and more heads as we go in lower resolution latent representations

## Classifier-Free Guidance

CIFAR-10, Euler method:

| <img src="src/images/euler/all_cifar10_10.png" width=250> | <img src="src/images/euler/all_cifar10_10_cfgscale_2_5.png" width=250> | <img src="src/images/euler/all_cifar10_10_cfgscale_5.png" width=250> | <img src="src/images/euler/all_cifar10_10_cfgscale_7.png" width=250> |
|:--:|:--:| :--:| :--:|
| <img src="src/images/euler/all_cifar10_90.png" width=250> | <img src="src/images/euler/all_cifar10_90_cfgscale_2_5.png" width=250> | <img src="src/images/euler/all_cifar10_90_cfgscale_5.png" width=250> | <img src="src/images/euler/all_cifar10_90_cfgscale_7.png" width=250> |
| *cfg.scale=1* | *cfg.scale=2.5* | *cfg.scale=5* | *cfg.scale=7* |

- Our **Classifier-Free Guidance** (CFG) scale $\alpha$ corresponds to using $\nabla_x \log p_{t, \alpha}(x|c) = (1-\alpha) \nabla_x \log p_t(x) + \alpha \nabla_x \log p_t(x|c)$ instead of the unconditional score function $\nabla_x \log p_t(x)$ of noisy marginal distributions in the original probability flow ODE $dx = -\dot{\sigma}(t) \sigma(t) \nabla_x \log p_t(x)dt$.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) paper uses, assuming we can directly replace a noise by a score function, $\nabla_x \log p_{t, \alpha}(x|c) = -\omega \nabla_x \log p_t(x) + (1+\omega) \nabla_x \log p_t(x|c)$ with $\omega=\alpha-1 > 0$. Therefore our CFG scale $\alpha$ should be greater than $1$.
- $\nabla_x \log p_{t, \alpha}(x|c)$ can also be rewritten as $\nabla_x \log p_t(x) + \alpha (\nabla_x \log p_t(x|c) - \nabla_x \log p_t(x))$
- The CFG scale $\alpha$ deforms the distribution and reduces diversity. As we can see from the generated CIFAR-10 pictures, a high CFG scale can cause saturated colors so one may opt for class-dependent scales.
- Our CFG scale is kept unchanged in our sampling methods.

# Credits

The computations were performed at University of Geneva using Baobab/Yggdrasil HPC service

- https://github.com/pytorch/pytorch
- https://github.com/crowsonkb/k-diffusion/tree/master
- https://wandb.ai
- https://github.com/zalandoresearch/fashion-mnist
- https://www.cs.toronto.edu/~kriz/cifar.html
- https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
