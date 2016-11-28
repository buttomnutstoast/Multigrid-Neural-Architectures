--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'optim'

--[[
    1. Create Model
    2. Create Criterion
    3. Convert model to CUDA
]]--

-- 1. Create Model
-- 1.1 inherit your customized net from basic net
local config = OPT.netType
BASICNETOBJ = paths.dofile('models/basic_model.lua')
NETOBJ = paths.dofile('models/' .. config .. '.lua')
setmetatable(NETOBJ, {__index=BASICNETOBJ})

-- 1.2 If preloading option is set, preload weights from existing models
-- appropriately
if OPT.retrain ~= 'none' then
    assert(paths.filep(OPT.retrain), 'File not found: ' .. OPT.retrain)
    print('Loading model from file: ' .. OPT.retrain);
    MODEL = loadDataParallel(OPT.retrain, OPT.nGPU, NETOBJ)
else
    print('=> Creating model from file: models/' .. config .. '.lua')
    MODEL = NETOBJ.createModel(OPT)
end

-- 2. Create Criterion
local origCriterion = NETOBJ.createCriterion()
if OPT.iterSize > 1 then
    CRITERION = nn.MultiCriterion()
    CRITERION:add(origCriterion, 1/OPT.iterSize)
else
    CRITERION = origCriterion
end

print('=> Model')
print(MODEL)

print('=> Criterion')
print(origCriterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
MODEL:cuda()
CRITERION:cuda()

collectgarbage()
