# Autoencoder Notebook

This repository is a small FashionMNIST autoencoder demo built around a single Jupyter notebook. It trains a simple feedforward autoencoder, shows how well it reconstructs images, and gives a quick look at the learned latent space.

## What’s inside

- A notebook that loads FashionMNIST from the local `data/` folder, or downloads it if the files are missing.
- A fully connected autoencoder for $28 \times 28$ grayscale images.
- Reconstruction plots so you can compare inputs with model output.
- Latent-space visualizations in 1D, 2D, 3D, or PCA form, depending on the latent size.
- A small sampling demo that decodes a random latent vector into an image.

## Notebook

- [autoencorder.ipynb](autoencorder.ipynb)

## Data

The notebook uses FashionMNIST by default:

- `data/FashionMNIST/raw/`

There are also MNIST files in `data/MNIST/raw/`, and the notebook includes a commented line you can switch on if you want to train on MNIST instead.

The notebook expects the `data/` folder to be available relative to the working directory, so make sure you open it from the project root.

## Requirements

You’ll need these Python packages:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `scikit-learn`

If CUDA is available, the notebook will use it automatically. Otherwise it runs on CPU.

## Model

The autoencoder is intentionally simple:

- Encoder: `784 -> 256 -> 128 -> latent_dim`
- Decoder: `latent_dim -> 128 -> 256 -> 784`

The default latent dimension is `64`.

## How to run it

1. Open [autoencorder.ipynb](autoencorder.ipynb).
2. Run the cells from top to bottom.
3. Let the model train for the configured number of epochs.
4. Check the reconstruction plot, latent-space visualization, and sampled output.

## What you should see

After the notebook finishes, it prints:

- The compute device in use.
- Training loss for each epoch.
- A grid of original and reconstructed images.
- A latent-space plot or PCA projection.
- A decoded image from a random latent vector.

## A couple of notes

- The notebook file is named `autoencorder.ipynb` right now. If you rename it later, update this README too.
- The comments already in the notebook are doing useful work, so I didn’t add extra ones.
- If you want to clean up the repo before pushing to GitHub, adding a `.gitignore` for notebook checkpoints would be a good next step.
