This package is extended from codes written by Soumith which provides easy ways to import and export dataset, which is used in training/testing/evaluating deep-net models by torch7.

The package is consist of following parts:

    1. Loading Command Line Options (opts.lua)
    2. Parallel Computations on Multi-GPUs (multigpu.lua)
    3. Deep-net Model Construction (model.lua)
    4. Parallel Data Loading (data.lua)
    5. Run Training/Testing/Evaluation (run.lua)

You should write your own codes in:

    1. models/YOURMODEL.lua
    2. donkey.lua and dataset.lua in dataset/YOURDATASET
    3. pipeline.lua, train.lua, test.lua and eval.lua in pipelines/YOURPIPELINE
**NOTE**: please make sure that

    1. functions including createModel, createCriterion, trainOutputInit,
       trainOutput, testOutputInit, testOutput, evalOutputInit, evalOutput,
       trainRule (or your should manually assing Learning Rate & Weight Decay)
       are implemneted in models/YOURMODEL.lua
    2. functions getInputs, genInput, get and sample are implemented in
       dataset/YOURDATASET/dataset.lua; getInputs and genInput should return
       table of input-data and table of ground-truth labels
    3. functions sampleHookTrain and sampleHoodTest are implemented in
       dataset/YOURDATASET/donkey.lua which explicitly load and process input
       data

There are global variables:

    1. MODEL (your own models)
    2. CRITERION (your own loss-criterion)
    3. NETOBJ (which defines how to create and optimize models)
    4. DONKEYS (parallel threads to load data)
    5. EPOCH (current epoch in training/testing/evaluating)
