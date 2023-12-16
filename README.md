# Diffusion
Deep Learning Project on Diffusion Models for Image Generation based on [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)'s paper by Karras et al.

## Generated samples

### Unconditional

**CelebA epoch 124, stochastic heun, 33M parameters U-Net model:**

| <img src="src/images/stochastic_heun/v1/uncond_samples_celeba_16_epoch_124.png" width=500> | <img src="src/images/stochastic_heun/v1/iterative_denoising_process_celeba_epoch_124.png" width=500> |
|:--:| :--:|
| *Randomly generated faces* | *An iterative denoising process* |

### Conditional generation with Classifier-Free Guidance

**FashionMNIST and CIFAR-10, 50 Euler method steps, 5M parameters U-Net model:**

| <img src="results/images/fashionmnist/euler/cond_10_cfgscale_1.png" width=500> | <img src="results/images/cifar10/euler/cond_10_cfgscale_2_5.png" width=500> |
|:--:| :--:|
| <img src="results/images/fashionmnist/euler/cond_90_cfgscale_1.png" width=500> | <img src="results/images/cifar10/euler/cond_90_cfgscale_2_5.png" width=500> |
| *FashionMNIST cfg.scale=1, 100 epochs* | *CIFAR-10 cfg.scale=2.5, 200 epochs* |

**Randomly generated horse and ship (cfg.scale=2.5, 50 Euler method steps):**


| <img src="results/images/cifar10/euler/iterative_denoising_process_class_7_cfgscale_2_5.png" width=500> | <img src="results/images/cifar10/euler/iterative_denoising_process_class_8_cfgscale_2_5.png" width=500> |
|:--:| :--:|


# How To Use?

## Training

If you want to train a model from scratch, please make sure to delete `checkpoints/checkpoint.pth`, otherwise training will resume.

```bash
python src/trainer.py
```

Note: if you specify `num_workers` > 0 in the Dataloader and get a warning like `UserWarning: This DataLoader will create 2 worker processes in total. Our suggested max number of worker in current system is 1, which is smaller than what this DataLoader is going to create`, you need to make sure that you have access to enough cpu cores. For instance in Unige Baobab/Yggdrasil HPC service, I have to specify `#SBATCH --cpus-per-task 4` in my Slurm "sbatch" script.

## Sampling

- Unconditional generation

```bash
python src/sampler.py
```

- Conditional generation with Classifier-Free Guidance (CFG) for a single class

```bash
python src/sampler.py common.sampling.label=<class-id> common.sampling.cfg_scale=<cfg-scale>
```

- Classifier-Free Guidance (CFG) for all classes

```bash
python src/sampler_all.py common.sampling.cfg_scale=1
```

You can change the sampling method by adding `common.sampling.method=<sampling-method>` where `<sampling-method>` can be either `euler`, `heun` or `stochastic_heun`.

<!-- # Tests (Work in Progress)

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
```-->

# Results

See `./results/README.md`.

# TODO list

- Analyze the effect of a different number of heads as we go in lower resolution latent representations
- Train EMA model and maybe use pixel unshuffle/shuffle
- Re-train everything with the same dataset splits. Training and validation sets are never the same across models due to the `random_split`!
- Recompute FIDs.

# Credits

The computations were performed at University of Geneva using Baobab/Yggdrasil HPC service

- https://github.com/pytorch/pytorch
- https://github.com/crowsonkb/k-diffusion/tree/master
- https://wandb.ai
- https://github.com/zalandoresearch/fashion-mnist
- https://www.cs.toronto.edu/~kriz/cifar.html
- https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
