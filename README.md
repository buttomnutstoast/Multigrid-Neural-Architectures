# Neural-MultiGrid
This implements the work from Neural-MultiGrid by Tsung-Wei Ke, Michael Maire and Stella Yu.

### Requirements

Please make sure that following libraries are well installed.

1. torch
2. cudnn v5 (or above)
3. cuda 7.5 (or above)

Also make sure you have installed following torch packages

1. nn
2. cutorch
3. cunn
4. cudnn
5. optim
6. hdf5
7. image
8. nccl

### Dataset preparation

1. Download [Cifar100-whitened](https://yadi.sk/d/em4b0FMgrnqxy)
2. Download MNIST-cluttered (we borrow and revise from [DeepMind's code](https://github.com/deepmind/mnist-cluttered))
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

Make sure that cifar and mnist data are located at `HOME_PREFIX/data/Cifar100-whitened/` and `HOME_PREFIX/data/mnist-cluttered/`
```
// Set the environment up
> export HOME_PREFIX=/path/to/dataset/rootdir
```

### Classification over Cifar100

```
> sh scripts/prnmg.sh (or vgg.sh, resnet.sh, nmg.sh, pnmg.sh, rnmg.sh, prnmg.sh)
```
##### Options:

* `-nLayer`: Number of conv or mg-conv layers in each block (set 1 for VGG-6/NMG-6/P-NMG-9/RES-12/R-NMG-12/PR-NMG-16, 2 for VGG-11/NMG-11/P-NMG-16/RES-22/R-NMG-22/PR-NMG-30, ... etc)


### Segmentation/Spatial-transformation over MNIST-cluttered
```
> sh scripts/unet.sh (or prnmg.mnist.sh, pnmg.mnist.sh)
```

##### Options:
* `-dataset`:  mnist-seg (segmentation), mnist-spt (spatial transformation), mnist-rot (pure rotation), mnist-sca (pure scaling), mnist-tra (pure translation), mnist-aff (pure affine transformation)

##### Testing:

Take segmentation for example, the fullpath of the trained models would be `checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/model_200.t7`.

First, add the following line in `scripts/mnist-seg.sh`
```
-retrain checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/model_200.t7
```
Second, change the option `-dataset` mnist-spt to mnist-seg in the script.

Last, run the testing code which will compute the meanIU and meanAcc of the predictions. Also, the predictions would be saved to `checkpoint/mnist-seg/MODEL_PREFIX/DATE_TIME/testOutput_1.h5`
```
> sh scripts/mnist-seg.sh
```

### Other Options
Modify the training scripts to customize your options

##### Batch size

* `-iterSize`: number of sub-iteration per iteration
* `-batchSize`: batch size of each sub-iteration

As a result, the real batch size = iterSize x batchSize

##### Multiple Gpus
* `-nGPU`

Please make sure that model return from `MODEL.lua` is `nn.DataParallelTable` (see [this](https://github.com/buttomnutstoast/Neural-MultiGrid/blob/master/models/prnmg.lua#L402-L406))

If you have more GPUs and want to save your time, set `nGPU`, `batchSize` larger and `iterSize` smaller, vice versa

##### For saliency map
* `-trainedNet`: path to trained network to render saliency map
