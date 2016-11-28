--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'os'
require 'utils/utilfuncs'

--[[
  1. Create loggers.
  2. train - this function handles the high-level training loop,
             i.e. load data, train model, save model and state to disk
  3. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Learning rate annealing schedule. We will build a optimizer for
-- each epoch.
--
-- Return values:
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if OPT.LR ~= 0.0 then -- if manually specified
      return {LR= OPT.LR, WD= OPT.weightDecay}
    else
        return NETOBJ.trainRule(epoch, OPT)
    end
end

-- 2. Create loggers.
local trainLogger = optim.Logger(paths.concat(OPT.save, 'train.log'))
local batchNumber
local currentvals, vals, tmpvals
local calltimes = 0

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. EPOCH)

    --[[
        fetching training information and options
    --]]
    local params = paramsForEpoch(EPOCH)
    optimState = {
        learningRate = params.LR,
        learningRateDecay = 0.0,
        momentum = OPT.momentum,
        dampening = 0.0,
        weightDecay = params.WD
    }
    batchNumber = 0
    calltimes = 0
    tmpvals = NETOBJ.trainOutputInit()
    currentvals = NETOBJ.trainOutputInit()
    vals = NETOBJ.trainOutputInit()
    cutorch.synchronize()

    -- set the dropouts to training mode
    MODEL:training()

    --[[
        loading data and training models
    --]]
    local tm = torch.Timer()
    for i=1,OPT.epochSize do
        for j=1, OPT.iterSize do
            local currentEpoch = EPOCH
            -- queue jobs to data-workers
            DONKEYS:addjob(
                -- the job callback (runs in data-worker thread)
                function()
                    local inputs, labels = trainLoader:genInputs(
                        OPT.batchSize,
                        currentEpoch
                        )
                    return inputs, labels
                end,
                -- the end callback (runs in the main thread)
                trainBatch
                )
        end
    end

    DONKEYS:synchronize()
    cutorch.synchronize()

    -- print information
    local strout = ('Epoch: [%d][TRAINING] Total Time(s): %.2f'):format(
        EPOCH, tm:time().real)
    local substrout = ''
    local loggerList = {}
    for k=1,#tmpvals do
        substrout = 'avg.'..tmpvals[k].name..':%.5f'
        loggerList['avg.'..tmpvals[k].name..' (train set)'] = (
            tmpvals[k].value / tmpvals[k].N
            )
        strout = strout..' '..substrout:format(tmpvals[k].value/tmpvals[k].N)
    end
    print(strout)
    print('\n')
    trainLogger:add(loggerList)

    collectgarbage()

end -- of train()
-------------------------------------------------------------------------------------------
local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = MODEL:getParameters()
local pa_noflat, gradPa_noflat = MODEL:parameters()

local inputsGPUTable = {}
local labelsGPUTable = {}
local inputs = nil
local labels = nil

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
    -- GPU inputs (preallocate)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    if calltimes==0 then timer:reset() end

    -- transfer over to GPU
    utilfuncs.put2GPU(inputsCPU, inputsGPUTable)
    utilfuncs.put2GPU(labelsCPU, labelsGPUTable)
    if #inputsGPUTable == 1 and torch.type(inputsGPUTable[1])~= 'table' then
        inputs = inputsGPUTable[1]
        inputsCPU = inputsCPU[1]
    else
        inputs = inputsGPUTable
    end
    if #labelsGPUTable == 1 and torch.type(labelsGPUTable[1])~= 'table' then
        labels = labelsGPUTable[1]
        labelsCPU = labelsCPU[1]
    else
        labels = labelsGPUTable
    end

    if calltimes==0 then
        MODEL:zeroGradParameters()
        currentvals = NETOBJ.trainOutputInit()
    end

    calltimes = calltimes + 1

    --[[
        forward and backward propogation
    --]]
    local outputs, err = NETOBJ.ftrain(inputs, labels, MODEL, CRITERION)
    cutorch.synchronize()

    NETOBJ.gradProcessing(MODEL, pa_noflat, gradPa_noflat, EPOCH)

    funcErrGradPa = function(x) return err, gradParameters end
    if calltimes == OPT.iterSize then
        NETOBJ.btrain(parameters, funcErrGradPa, optim, optimState)
        -- DataParallelTable's syncParameters
        if MODEL.needsSync then
            MODEL:syncParameters()
        end
    end
    cutorch.synchronize()

    do
        NETOBJ.trainOutput(vals, outputs, labelsCPU, err, OPT.iterSize)
        for k = 1,#vals do
            currentvals[k].value = currentvals[k].value+vals[k].value*vals[k].N
            currentvals[k].N     = currentvals[k].N+vals[k].N
        end
    end

    if calltimes == OPT.iterSize then
        batchNumber = batchNumber + 1

        -- print information
        local strout =
            ('%s Epoch: [%d][%d/%d]\tRun:%.3fs lr:%.3e Data:%.3fs'):format(
                os.date("%x %X"), EPOCH, batchNumber,
                OPT.epochSize, timer:time().real,
                optimState.learningRate, dataLoadingTime
                )
        local substrout = ''
        for k=1,#currentvals do
            substrout = currentvals[k].name..':%.5f'
            strout = strout .. ' ' .. substrout:format(
                currentvals[k].value / currentvals[k].N
                )
            tmpvals[k].value = tmpvals[k].value + currentvals[k].value
            tmpvals[k].N = tmpvals[k].N + currentvals[k].N
        end
        print(strout)

        dataTimer:reset()
        calltimes = 0
    end

    inputs = nil
    labels = nil
end
