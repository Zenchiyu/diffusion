# Diffusion
Deep Learning Project on Diffusion Models for Image Generation

# How To Use?

## Training

```bash
python src/trainer.py
```


## Sampling

- Unconditional generation

```bash
python src/sampler.py
```

- Classifier-Free Guidance (CFG)

Example
```bash
python src/sampler.py common.sampling.label=7 common.sampling.cfg_scale=0.8
```
