# Variational Autoencoder on MNIST

This repository is a small VAE demo built around one Jupyter notebook. It trains a fully connected Variational Autoencoder on MNIST, shows the loss during training, and gives you a quick way to inspect generated or reconstructed digits.

## Notebook

- [vae.ipynb](vae.ipynb)

## What’s in the notebook

The notebook walks through the full VAE workflow step by step:

- loading MNIST with PyTorch and `torchvision`
- preparing the images with a small transform pipeline
- defining a multilayer perceptron VAE
- applying the reparameterization trick
- training with reconstruction loss plus KL divergence
- printing the loss after each epoch

## Model overview

The model is intentionally simple:

- Input dimension: `784` for flattened $28 \times 28$ grayscale images
- Encoder:
  - `784 -> 256`
  - `256 -> 128`
  - splits into `mu` and `logvar`
- Latent dimension: `32`
- Decoder:
  - `32 -> 128`
  - `128 -> 256`
  - `256 -> 784`

The hidden layers use ReLU activations, and the decoder ends with a sigmoid output layer.

## Loss function

The training objective combines two parts:

- reconstruction loss: `F.mse_loss(..., reduction='sum')`
- KL divergence: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`

The final loss is the sum of both terms.

## Data

The notebook downloads and uses MNIST from:

- `../data`

If the dataset is not already present, PyTorch will download it automatically the first time you run the notebook.

The current transform pipeline is:

- `ToTensor()`
- `Normalize((0.5,), (0.5,))`

## Requirements

Install these Python packages:

- `torch`
- `torchvision`
- `matplotlib`

If you plan to extend the notebook later, `numpy` is useful but not required for the current cells.

## How to run it

1. Open [vae.ipynb](vae.ipynb).
2. Run the cells from top to bottom.
3. Make sure the `../data` path is valid from your current working directory.
4. Run the remaining cells in order.
5. Watch the loss values print after each epoch.

## What to expect

When you run the notebook, you should see:

- the selected compute device
- the dataset download or load step
- a VAE model initialized on CPU or GPU
- epoch-by-epoch loss values

## Why the loss can look large

The loss can look bigger than you might expect for two reasons:

- The reconstruction term uses `reduction='sum'`, so it adds error over all pixels and all samples in a batch.
- The input transform normalizes the data, but the decoder ends with `sigmoid()`, which outputs values in `[0, 1]`. That mismatch can make reconstruction harder and can keep the loss high.

If you want a more stable and easier-to-read loss value, try one of these adjustments:

- remove normalization and keep `sigmoid()` in the decoder
- keep normalization and change the decoder output activation to `tanh()`
- switch the reconstruction loss from `sum` to `mean`

## Project structure

- `vae.ipynb` - main notebook
- `README.md` - project overview and setup notes

## Notes

- This notebook is intentionally simple and is meant as a learning implementation rather than a production VAE.
- The model trains on flattened image vectors rather than convolutional features.
- If you plan to push this to GitHub, adding a `.gitignore` for notebook checkpoints and generated files is recommended.
