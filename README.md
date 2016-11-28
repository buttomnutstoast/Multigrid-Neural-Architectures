# Neural-MultiGrid
This implements the work from Neural-MultiGrid by Tsung-Wei Ke, Michael Maire and Stella Yu.

### Requirements

Please make sure that following libraries are well installed.

1. torch
2. cudnn v4 or v5
3. cuda 7.5

Also make sure you have installed following torch packages

1. nn
2. cutorch
3. cunn
4. cudnn
5. optim
6. hdf5
7. image
8. nccl

### Dataset

1. [Cifar100-whitened](https://yadi.sk/d/em4b0FMgrnqxy)
2. MNIST-cluttered (we borrow and revise from [DeepMind's code](https://github.com/deepmind/mnist-cluttered))
```
> cd utils/mnist-cluttered
> th download_mnist.lua

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

### Train the model

For the following steps, we assume that you will put cifar and mnist data under `HOME_PREFIX/data/Cifar100-whitened/` and `HOME_PREFIX/data/mnist-cluttered/`
```
// Set the environment up
> export HOME_PREFIX=/path/to/dataset/rootdir

// Train the model
> sh scripts/prnmg.sh
```

### Options
Modify the training scripts to customize your options

##### Batch size

* `-iterSize`: number of sub-iteration per iteration
* `-batchSize`: batch size of each sub-iteration

As a result, the real batch size = iterSize x batchSize

##### Multiple Gpus
* `-nGPU`

Please make sure that model return from `MODEL.lua` is `nn.DataParallelTable` (see [this](https://github.com/buttomnutstoast/Neural-MultiGrid/blob/master/models/prnmg.lua#L402-L406))

If you have more GPUs and want to save your time, set `nGPU`, `batchSize` larger and `iterSize` smaller, vice versa

##### For VGG, NMG, P-NMG, R-NMG, PR-NMG over Cifar100
* `-nLayer`: Number of conv or mg-conv layers in each block

##### For saliency map
* `-trainedNet`: path to trained network to render saliency map
