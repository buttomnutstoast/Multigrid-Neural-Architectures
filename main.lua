--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--[[
    The package is consist of following parts:
    1. Loading Command Line Options (opts.lua)
    2. Parallel Computations on Multi-GPUs (multigpu.lua)
    3. Deep-net Model Construction (model.lua)
    4. Parallel Data Loading (data.lua)
    5. Training/Testing/Evaluation (train.lua/test.lua/eval.lua)

    You should write your own codes in:
    1. models/YOURMODEL.lua
    2. donkey.lua and dataset.lua in dataset/YOURDATASET/

    There are global variables:
    1. MODEL (your own models)
    2. OPT (environment options)
    3. CRITERION (your own loss-criterion)
    4. NETOBJ (which defines how to create and optimize models)
    5. DONKEYS (parallel threads to load data)
    6. EPOCH (current epoch in training/testing/evaluating)
--]]
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

-- load options
local opts = paths.dofile('opts.lua')

OPT = opts.parse(arg)
print(OPT)
torch.manualSeed(OPT.manualSeed)
print('Saving everything to: ' .. OPT.save)
os.execute('mkdir -p ' .. OPT.save)

-- import required functions for multi-GPU
paths.dofile('multigpu.lua')

-- create customized model
paths.dofile('model.lua')
cutorch.setDevice(OPT.GPU) -- by default, use GPU 1

-- import dataset
paths.dofile('data.lua')

-- run training/testing/evaluation pipelines
paths.dofile('run.lua')