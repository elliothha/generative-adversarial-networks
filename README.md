# Generative Adversarial Networks (GANs) for Generative Modeling
![GitHub last commit](https://img.shields.io/github/last-commit/elliothha/generative-adversarial-networks) ![GitHub repo size](https://img.shields.io/github/repo-size/elliothha/generative-adversarial-networks)

*[3/16/24 Update] Uploaded old notebook file for DCGAN implementation*

This repo contains a PyTorch implementation of a Deep Convolutional GAN (DCGAN) used to model the MNIST dataset. Purely intended for educational purposes.

Results after training found [here](https://github.com/elliothha/generative-adversarial-networks/tree/main?tab=readme-ov-file#after-30-training-epochs). Full generative modeling zoo found [here](https://github.com/elliothha/generative-modeling-zoo).

by **Elliot H Ha**. Duke University

[elliothha.tech](https://elliothha.tech/) | [elliot.ha@duke.edu](mailto:elliot.ha@duke.edu)

---

## Dependencies
- Jupyter Notebook
- PyTorch

## Project Structure
`models/` is the main folder containing the Jupyter Notebook file implementing the DCGAN model for the MNIST dataset. The raw dataset is stored in `models/data/MNIST/raw`.

## Hyperparameters & Architecture
```
p = 0.3 # dropout for the Discriminator
lr = 2e-4
betas = (0.5, 0.999)
num_epochs = 30
batch_size = 200

z_dim = 100 # input noise dimensionality for the Generator
```

I use [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) as my optimizer with a learning rate, `lr = 2e-4`, `betas = (0.5, 0.999)`, default epsilon, and 0 weight decay. 

### Generator Architecture
The Generator is implemented with 4 sequential "Transposed Convolution (TC) blocks", with each block consisting of a 2D Transposed Convolutional layer, a 2D Batch Norm layer, and then an in-place ReLU activation. After the last block (TC3), a singular 2D Transposed Convolutional layer is used with a Tanh activation afterwaards.

Generator = `[TC0, TC1, TC2, TC3, nn.ConvTranspose2d(), nn.Tanh()]`

Specific channel/feature parameters and convolution parameters can be found within the Notebook file.

Additionally, Convolutional layers are initialized using a normal distribution with `mean=0.0`, `std=0.02`. BatchNorm layers are initialized using a normal distribution with `mean=1.0`, `std=0.02`, and `biases=0`.

### Discriminator Architecture
The Discriminator is implemented with "Convolution (C) blocks" and "MaxPool (MP) blocks", with each "Convolution block" consisting of a 2D Convolutional layer, a 2D Batch Norm layer, and then an in-place **Leaky** ReLU activation. Each "MaxPool block" consists of a 2D MaxPool layer, and then a Dropout layer with `p = 0.3`. Finally, the output is flattened and passed through two Linear layers with a Sigmoid activation afterwards.

Discriminator = `[C1, MP1, C2, MP2, C3, MP3, C4, nn.Flatten(), nn.Linear(), nn.Linear(), nn.Sigmoid()]`

Specific channel/feature parameters and convolution parameters can be found within the Notebook file.

### DCGAN Architecture

The DCGAN initializes both a Generator and a Discriminator, with separate Adam optimizers for both. 

Training is run for `num_epochs = 30` epochs with a `batch_size = 200`.

## DCGAN Generated Sample Results
### After 30 Training Epochs
Discriminator Loss: 98.0899, Generator Loss: 98.1262
![DCGAN sampling results for 30 training epochs](/examples/samples.png)

---

## References
1. Avinash Hindupur's GitHub repo for keeping track of "all named GANs" | [link](https://github.com/hindupuravinash/the-gan-zoo)
2. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*, Radford et. al 2015 | [1511.06434](https://arxiv.org/abs/1511.06434)
3. Implementation of the DCGAN by Radford et. al | [link](https://github.com/Newmu/dcgan_code)
4. *Generative Adversarial Networks*, Goodfellow et. al 2014 | [1406.2661](https://arxiv.org/abs/1406.2661)
5. Implementation of the GAN by Goodfellow et. al | [link](https://github.com/goodfeli/adversarial)
6. Zak Jost's "Gentle Intro to Generative Adversarial Networks - Part 1 (GANs)" | [link](https://www.youtube.com/watch?v=3z8VSpBL6Vg)
7. Zak Jost's "Understand the Math and Theory of GANs in ~ 10 minutes" | [link](https://www.youtube.com/watch?v=J1aG12dLo4I&t=1s)

I found Zak Jost's videos on GANs *extremely* helpful as he walks through the derivation step by step, which cleared things up for me the most.
