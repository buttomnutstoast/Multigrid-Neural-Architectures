# Multigrid Neural Architecture
This implements the work from [Multigrid Neural Architecture](https://arxiv.org/abs/1611.07661) by [Tsung-Wei Ke](https://www1.icsi.berkeley.edu/~twke/), [Michael Maire](http://ttic.uchicago.edu/~mmaire/) and [Stella Yu](https://www1.icsi.berkeley.edu/~stellayu/).

![multigrid_cnn](/figures/multigrid_cnn.png)

We propose a multigrid extension of convolutional neural networks (CNNs). Rather than manipulating representations living on a single spatial grid, our network layers operate across scale space, on a pyramid of grids. They consume multigrid inputs and produce multigrid outputs; convolutional filters themselves have both within-scale and cross-scale extent. This aspect is distinct from simple multiscale designs, which only process the input at different scales. Viewed in terms of information flow, a multigrid network passes messages across a spatial pyramid. As a consequence, receptive field size grows exponentially with depth, facilitating rapid integration of context. Most critically, multigrid structure enables networks to learn internal attention and dynamic routing mechanisms, and use them to accomplish tasks on which modern CNNs fail.

We demonstrate the network by running our model on different tasks: 1) Classification on cifar100 & ImageNet, 2) Segmentation on cluttered Mnist data, 3) Spatial Transformation on cluttered Mnist data. The following figure shows the multigrid convolutional and multigrid residual convolutional layer used in our network.

![multigrid_cnn](/figures/multigrid_layers.png)


## Prerequisites

Please make sure that following libraries are well installed.

* Torch7
* Linux
* NVIDIA GPU + CUDA + cuDNN

## Getting Started

* Install [Torch and dependencies](https://github.com/torch/distro)
* Install Torch packages `nn`, `cutorch`, `cunn`, `cudnn`, `optim`, `hdf5`
```
> luarocks install nn
> luarocks install cutorch
> luarocks install cunn
> luarocks install cudnn
> luarocks install optim
> luarocks install hdf5
> luarocks install image
```

* Install `nccl`, if you want to acclerate computations when using multiple GPUs. Compile and install the [library](https://github.com/NVIDIA/nccl) first, then:
```
> luarocks install nccl
```


## Set up for Environment Path
```
> export HOME_PREFIX=/path/to/dataset/rootdir
```


## Prepare for dataset

* Download `cifar100_whitened.t7` from [Here](https://yadi.sk/d/em4b0FMgrnqxy), put the .t7 file under `$HOME_PREFIX/data/Cifar100-whitened/`.
* Generate datas for MNIST-cluttered (we borrow and revise from [DeepMind's code](https://github.com/deepmind/mnist-cluttered))
```
> cd utils/mnist-cluttered
> th download_mnist.lua

// Generating data & labels:
// For segmentation
> th segmentation.lua
// For spatial transformer
> th spatial_transform.lua
// For pure rotation
> th rotation.lua
// For pure affine transformation
> th affine_transform.lua
// For pure translation
> th translation.lua
```

Make sure that all .t7 files for mnist-cluttered data are put under `$HOME_PREFIX/data/mnist-cluttered/`.


## Training for Cifar100

```
> sh scripts/cifar/prnmg.sh (or vgg.sh, resnet.sh, nmg.sh, pnmg.sh, rnmg.sh, prnmg.sh)
```

#### Options:

* `-nLayer`: Number of conv or mg-conv layers in each block (set 1 for VGG-6/NMG-6/P-NMG-9/RES-12/R-NMG-12/PR-NMG-16, 2 for VGG-11/NMG-11/P-NMG-16/RES-22/R-NMG-22/PR-NMG-30, ... etc)

***Top-1 prediction error over cifar100***:

Network | Params(x10^6) | FLOPs(x10^6) | Error (%)
--------|---------------|--------------|----------
MG-6 | 8.34 | 116.63 | 32.08
MG-11 | 20.46 | 391.88 | 28.39
MG-16 | 32.58 | 667.13 | 29.91
MG-21 | 44.68 | 942.38 | 30.03
R-MG-12 | 20.56 | 457.20 | 27.84
R-MG-22 | 44.79 | 1007.70 | 26.79
R-MG-32 | 69.02 | 1558.20 | 25.29
R-MG-42 | 93.26 | 2108.71 | 26.32

## Training for ImageNet

```
> sh scripts/ilsvrc/rnmg.sh (or prnmgseg.sh)
```

#### options:
* `-nGPU`: Number of GPU used for training, set to 1 if you have onlye one GPU available (which is very slow).

***Top-1 prediction error over validation set of ImageNet***:

Network | Params(x10^6) | FLOPs(x10^9) | val, 10-crop (%) | val, single-crop (%)
--------|---------------|--------------|------------------|-----------------
ResNet-50 | 25.6 | 4.46 | 22.85 | -
WRN-34 (2.0) | 48.6 | 14.09 | - | 24.50
R-MG-34 | 32.9 | 5.76 | 22.42 | 24.51



## Segmentation/Spatial-transformation over MNIST-cluttered

```
> sh scripts/mnist-cluttered/unet.sh (or prnmg.mnist.sh, pnmg.mnist.sh)
```

#### Options:
* `-dataset`:  mnist-seg (segmentation), mnist-spt (spatial transformation), mnist-rot (pure rotation), mnist-sca (pure scaling), mnist-tra (pure translation), mnist-aff (pure affine transformation)

### Testing for segmentation/spatial-transformation:

Take segmentation for example, the fullpath of the trained models would be `checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/model_200.t7`.

1. Add the following line to `scripts/mnist-cluttered/mnist-test.sh`
```
-retrain checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/model_200.t7
```
2. Change the option `-dataset` mnist-spt to mnist-seg in the script.

3. Run the script to compute the meanIU and meanAcc of the predictions. Also, the predictions would be saved to `checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/testOutput_1.h5`
```
> sh scripts/mnist-test.sh
```

***Spatial transformation over cluttered MNIST***:
![spatial_transform](/figures/spatial_transform.png)

### Generating Saliency Map

1. Add the following line to `scripts/mnist-cluttered/mnist-saliency.sh`
```
-trainedNet checkpoint/mnist-spt/MODEL_PREFIX/DATE_TIME/model_200.t7
```

2. Run the script

***Saliency Map of Unet/MG***:
![saliency_map](/figures/saliency_map.png)
