th main.lua \
-data $HOME_PREFIX/data/mnist-cluttered \
-dataset mnist-saliency \
-nDonkeys 1 \
-nEpochs 1 \
-epochSize 150 \
-batchSize 1 \
-colorspace bgr \
-netType mnist-saliency \
-pipeline saliency \
-nGPU 2 \
-test

