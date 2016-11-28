local NET = {}
function NET.packages()
    require 'optnet'
    require 'cudnn'
    require 'cunn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

require 'cudnn'
local Convolution = cudnn.SpatialConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization
local UpSample = nn.SpatialUpSamplingNearest

-----------------
-- Basic Units --
-----------------
local function ConvBNReLU(mod, nIP, nOP, kernel, dropout)
    local k = kernel
    local p = k == 1 and 0 or 1

    if dropout and dropout > 0 then
        mod:add(nn.Dropout(dropout))
    end
    mod:add(Convolution(nIP, nOP, k, k, 1, 1, p, p))
    mod:add(SBatchNorm(nOP, 1e-3))
    mod:add(ReLU(true))
    return mod
end

local function ResampleConcat(nIPs)
    -- helper function to output module which perform
    -- resampling, followed by concatenation
    local resample_concat = nn.ConcatTable()
    local nOPs = {}

    local nGrids = #nIPs
    for iG = 1,nGrids do
        -- resampling
        local grid = nn.Sequential()
        local multi_scales = nn.ConcatTable()
        local nIP = 0
        -- 1. down sampling from previous finer grid
        if iG-1 > 0 then
            local finer_scale = nn.Sequential()
            finer_scale:add(nn.SelectTable(iG-1))
            finer_scale:add(Max(2, 2, 2, 2, 0, 0):ceil())
            multi_scales:add(finer_scale)

            nIP = nIPs[iG-1] + nIP
        end

        -- 2. no scale-resizing at this grid
        local same_scale = nn.SelectTable(iG)
        multi_scales:add(same_scale)

        nIP = nIPs[iG] + nIP

        -- 3. up sampling from coarser grid
        if iG+1 <= nGrids then
            local coarser_scale = nn.Sequential()
            coarser_scale:add(nn.SelectTable(iG+1))
            coarser_scale:add(UpSample(2))
            multi_scales:add(coarser_scale)

            nIP = nIPs[iG+1] + nIP
        end

        grid:add(multi_scales)

        -- concatenation
        grid:add(nn.JoinTable(2))

        resample_concat:add(grid)
        nOPs[iG] = nIP
    end

    return resample_concat, nOPs
end

local function mgConv(nInputPlanes, nOutputPlanes, kernels, dropout)
    -- Build a module which take multi-scale grids as input and output another
    -- multi-scale grids:
    --    inputGrids: {input_scale_0, input+scale_1, ..., input_scale_n}
    --    outputGrids: {output_scale_0, outptu_scale_1, ..., output_scale_n}
    -- Args:
    --    nInputPlanes: dimension of input grids
    --    nOutputPlanes: dimension of output grids
    --    kernels: kernel size of each grid
    --    dropout: droptoub rate
    assert(#nInputPlanes == #nOutputPlanes,
        'number of input grid should be equal to output grid')
    assert(#nInputPlanes == #kernels,
        'should provide kernel size for every scale of grid')

    local mg_conv = nn.Sequential()
    local resample_concat, _nIPs = ResampleConcat(nInputPlanes)
    mg_conv:add(resample_concat)

    local convs = nn.ParallelTable()
    for i = 1,#_nIPs do
        local _conv = nn.Sequential()
        ConvBNReLU(_conv, _nIPs[i], nOutputPlanes[i], kernels[i], dropout)
        convs:add(_conv)
    end
    mg_conv:add(convs)

    return mg_conv
end

local function mgConvInput(nOutputPlanes)
    -- build spatial pyramids of input image at different scales
    -- Args:
    --    ratio: length ratio between each grid and its next-scale grid
    local mg_inputs = nn.ConcatTable()
    for iG = 1,#nOutputPlanes do
        local proc = nn.Sequential()
        if iG == 1 then
            proc:add(nn.Identity())
        else
            local r = torch.pow(2, iG-1)
            local resize = Avg(r, r, r, r, 0, 0)
            proc:add(resize)
        end
        ConvBNReLU(proc, 3, nOutputPlanes[iG], 3)
        mg_inputs:add(proc)
    end
    return mg_inputs
end

local function mgPool(nInputPlanes, isConcat)
    local mg_pool = nn.ConcatTable()
    -- max-pool or concatenate
    local nGrids = #nInputPlanes
    for i = 1,nGrids do
        local proc = nn.Sequential()
        if i == nGrids-1 and isConcat then
            local pool_cat = nn.ConcatTable()
            local pool = nn.Sequential()
                :add(nn.SelectTable(i))
                :add(Max(2,2,2,2,0,0):ceil())
            local cat = nn.SelectTable(i+1)

            pool_cat:add(pool)
            pool_cat:add(cat)
            proc:add(pool_cat)
            proc:add(nn.JoinTable(2))

            -- update nInputPlanes
            nInputPlanes[i] = nInputPlanes[i] + nInputPlanes[i+1]
            nInputPlanes[i+1] = nil
        else
            proc:add(nn.SelectTable(i))
            proc:add(Max(2,2,2,2,0,0):ceil())
        end
        mg_pool:add(proc)
        
        if i == nGrids-1 and isConcat then
            break
        end
    end

    return mg_pool
end

----------------------
-- helper functions --
----------------------
local function update(old, new)
    assert(#old == #new)
    for i = 1,#new do old[i] = new[i] end
end

local function MultiGridsInput(model, nIPs, nOPs, nLayer, dropout)
    -- always perform
    -- <mg-conv-input> x 1
    -- <conv> x nLayer (3rd grid)
    -- <mg-conv> x nLayer (2nd-3rd grid)
    -- <mg-conv> x nLayer (1st-2nd-3rd grid)

    -- <mg-conv-input>
    model:add(mgConvInput(nOPs))
    update(nIPs, nOPs)

    for nGrid = 1,#nOPs do
        if nGrid > 1 then
            -- <mg-conv>
            for i = 1,nLayer do
                local mg_convs = nn.ConcatTable()
                for j = 1,#nOPs-nGrid do
                    mg_convs:add(nn.SelectTable(j))
                end

                local _mg_conv = nn.Sequential()
                local _select = nn.ConcatTable()
                local _nOPs = {}
                local _kernels = {}
                for j = #nOPs-nGrid+1, #nOPs do
                    _select:add(nn.SelectTable(j))
                    _nOPs[#_nOPs+1] = nOPs[j]
                    _kernels[#_kernels+1] = 3
                end
                _mg_conv:add(_select)

                _mg_conv:add(mgConv(_nOPs, _nOPs, _kernels, dropout))
                mg_convs:add(_mg_conv)
                model:add(mg_convs)
                model:add(nn.FlattenTable())
            end
        else
            -- <conv>
            for i = 1,nLayer do
                local convs = nn.ParallelTable()
                for j = 1,#nOPs-1 do
                    convs:add(nn.Identity())
                end

                local _conv = nn.Sequential()
                ConvBNReLU(_conv, nOPs[#nOPs], nOPs[#nOPs], 3, dropout)
                convs:add(_conv)
                model:add(convs)
            end
        end
    end
end

local function MultiGrids(model, nIPs, nOPs, kernels, nLayer, dropout)
    -- <mg-conv> x nLayer
    for i = 1,nLayer do
        model:add(mgConv(nIPs, nOPs, kernels, dropout))
        update(nIPs, nOPs)
    end
end

function NET.createModel(opt)
    NET.packages()

    local model = nn.Sequential()

    local blocks = {
        {{64,32,16}, {3,3,3}},
        {{128,64,32}, {3,3,3}},
        {{256,128,64}, {3,3,3}},
        {{512,256,128}, {3,3,1}},
        {{512,384}, {3,1}},
    }
    local dropouts = {nil,0.1,0.2,0.3,0.4}


    local nIPs = {3,3,3}
    local nOPs
    for indBlock = 1,#blocks do
        nOPs = blocks[indBlock][1]
        local kernels = blocks[indBlock][2]
        local dropout = opt.isDropout and dropouts[indBlock]

        if indBlock == 1 then
            MultiGridsInput(model, nIPs, nOPs, opt.nLayer, dropout)
        else
            MultiGrids(model, nIPs, nOPs, kernels, opt.nLayer, dropout)
        end

        local isConcat = kernels[#kernels] == 1 and true or false
        model:add(mgPool(nIPs, isConcat))
    end

    local nLinear
    if opt.dataset == 'cifar10' then
        nLinear = 10
    else
        nLinear = 100
    end

    local classifier = nn.Sequential()
    classifier:add(nn.SelectTable(1))
    classifier:add(nn.View(-1, nIPs[1]))
    classifier:add(nn.Linear(nIPs[1], nLinear))
    classifier:add(nn.LogSoftMax())
    model:add(classifier)

    -- initialization from MSR
    local function MSRinit(net)
        local function init(name)
            for k,v in pairs(net:findModules(name)) do
                local n = v.kW*v.kH*v.nOutputPlane
                v.weight:normal(0,math.sqrt(2/n))
                v.bias:zero()
            end
        end
        -- have to do for both backends
        init('nn.SpatialConvolution')
        init('cudnn.SpatialConvolution')
    end

    MSRinit(model)

    if opt.nGPU > 1 then
        return makeDataParallel(model, opt.nGPU, NET)
    else
        return model
    end
end

function NET.createCriterion()
    local criterion = nn.MultiCriterion()
    criterion:add(nn.ClassNLLCriterion())
    return criterion
end

function NET.trainOutputInit()
    local info = {}
    -- utilfuncs.newInfoEntry is defined in utils/train_eval_test_func.lua
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err, iterSize)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()

    info[1].value   = err * iterSize
    info[1].N       = batch_size

    info[2].value   = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N       = batch_size
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size

    info[2].value = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N     = batch_size
end

function NET.trainRule(currentEpoch, opt)
    local delta = 3
    local start = 1 -- LR: 10^-(star) ~ 10^-(start + delta)
    local ExpectedTotalEpoch = opt.nEpochs
    return {LR= 10^-((currentEpoch-1)*delta/(ExpectedTotalEpoch-1)+start),
           WD= 5e-4}
end

function NET.arguments(cmd)
    cmd:option('-nLayer', 1, 'number of layers per block')
    cmd:option('-isDropout', false, 'if using dropout')
end

return NET
