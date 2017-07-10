# Multigrid Neural Architecture
This implements the work from Multigrid Neural Architecture by [Tsung-Wei Ke](https://www1.icsi.berkeley.edu/~twke/), [Michael Maire](http://ttic.uchicago.edu/~mmaire/) and [Stella Yu](https://www1.icsi.berkeley.edu/~stellayu/).

### Prerequisites

Please make sure that following libraries are well installed.

* Torch7
* Linux
* NVIDIA GPU + CUDA + cuDNN

### Getting Started

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

### Prepare for dataset 

* Download `cifar100_whitened.t7` from [Here](https://yadi.sk/d/em4b0FMgrnqxy), put the .t7 file under `HOME_PREFIX/data/Cifar100-whitened/`.
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

Make sure that all .t7 files for mnist-cluttered data are put under `HOME_PREFIX/data/mnist-cluttered/`.

### Set up for Environment Path
```
> export HOME_PREFIX=/path/to/dataset/rootdir
```

### Training for Cifar100

```
> sh scripts/cifar/prnmg.sh (or vgg.sh, resnet.sh, nmg.sh, pnmg.sh, rnmg.sh, prnmg.sh)
```

##### Options:

* `-nLayer`: Number of conv or mg-conv layers in each block (set 1 for VGG-6/NMG-6/P-NMG-9/RES-12/R-NMG-12/PR-NMG-16, 2 for VGG-11/NMG-11/P-NMG-16/RES-22/R-NMG-22/PR-NMG-30, ... etc)

### Training for ImageNet

```
> sh scripts/ilsvrc/rnmg.sh (or prnmgseg.sh)
```

#### options:
* `-nGPU`: Number of GPU used for training, set to 1 if you have onlye one GPU available (which is very slow).


### Segmentation/Spatial-transformation over MNIST-cluttered

```
> sh scripts/mnist-cluttered/unet.sh (or prnmg.mnist.sh, pnmg.mnist.sh)
```

##### Options:
* `-dataset`:  mnist-seg (segmentation), mnist-spt (spatial transformation), mnist-rot (pure rotation), mnist-sca (pure scaling), mnist-tra (pure translation), mnist-aff (pure affine transformation)

##### Testing for segmentation/spatial-transformation:

Take segmentation for example, the fullpath of the trained models would be `checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/model_200.t7`.

1. Add the following line to `scripts/mnist-cluttered/mnist-seg.sh`
```
-retrain checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/model_200.t7
```
2. Change the option `-dataset` mnist-spt to mnist-seg in the script.

3. Run the script to compute the meanIU and meanAcc of the predictions. Also, the predictions would be saved to `checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/testOutput_1.h5`
```
> sh scripts/mnist-seg.sh
```

##### Generating Saliency Map

1. Add the following line to `scripts/mnist-cluttered/mnist-saliency.sh`
```
-trainedNet checkpoint/mnist-spt/MODEL_PREFIX/DATE_TIME/model_200.t7
```

2. Run the script
