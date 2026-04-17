# Vision Transformer for CIFAR-10

This notebook shows a simple Vision Transformer, or ViT, that learns to look at tiny pieces of pictures and guess what the picture is.

## What this project does

- loads the CIFAR-10 picture set
- changes pictures into tensors, which are number versions of images
- cuts each image into small patches
- sends the patches through transformer blocks
- makes a final class prediction
- checks accuracy on the test set
- shows a grid of predictions

## How the model works

Think of the image like a puzzle.

1. The image is turned into numbers.
2. The image is split into small squares called patches.
3. Each patch is sent into the transformer.
4. The transformer looks at how the patches connect with each other.
5. The model uses one special class token to make the final guess.

## Main parts in the notebook

- PatchEmbedding: turns image patches into vectors the model can read
- TransformerEncoderBlock: helps patches pay attention to each other
- MLPHead: turns the final hidden vector into class scores
- ViT: joins all parts together

## What each notebook section does

- Step 1: loads and prepares the images
- Step 2: makes mini-batches with DataLoader
- Step 3: sets the model size and training values
- Step 4: splits images into patches
- Step 5: lets the model compare patches
- Step 6: picks the class label
- Step 7: builds the full ViT model
- Step 8: creates the device, loss function, and optimizer
- Step 9: trains the model
- Step 10: checks accuracy on validation data
- Step 11: shows sample predictions

## How to run it

Open vit.ipynb and run the cells from top to bottom.

If the CIFAR-10 data is not already there, the notebook will try to download it into the data folder.

## Simple idea

The notebook teaches the computer to look at small parts of a picture, compare them, and guess the right label.