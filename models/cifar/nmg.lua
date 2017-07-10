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

local function ConvBNReLU(mod, nIP, nOP, kernel, dropout)
    local k = kernel
    local p = k == 1 and 0 or 1

    mod:add(Convolution(nIP, nOP, k, k, 1, 1, p, p))
    mod:add(SBatchNorm(nOP, 1e-3))
    mod:add(ReLU(true))
    if dropout and dropout > 0 then
        mod:add(nn.Dropout(dropout))
    end
    return mod
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

    local nGrids = #nInputPlanes
    local multi_grids = nn.ConcatTable()
    for iG = 1,nGrids do
        local grid = nn.Sequential()
        local multi_scales = nn.ConcatTable()
        local nInput = 0
        -- down sampling from previous finer grid
        if iG-1 > 0 then
            local finer_scale = nn.Sequential()
            finer_scale:add(nn.SelectTable(iG-1))
            finer_scale:add(Max(2, 2, 2, 2, 0, 0):ceil())
            multi_scales:add(finer_scale)

            nInput = nInputPlanes[iG-1] + nInput
        end

        -- no scale-resizing at this grid
        local same_scale = nn.SelectTable(iG)
        multi_scales:add(same_scale)

        nInput = nInputPlanes[iG] + nInput

        -- up sampling from coarser grid
        if iG+1 <= nGrids then
            local coarser_scale = nn.Sequential()
            coarser_scale:add(nn.SelectTable(iG+1))
            coarser_scale:add(UpSample(2))
            multi_scales:add(coarser_scale)

            nInput = nInputPlanes[iG+1] + nInput
        end

        grid:add(multi_scales)
        grid:add(nn.JoinTable(2)) -- concat multi-scale grids at dimension 2
        ConvBNReLU(grid, nInput, nOutputPlanes[iG], kernels[iG], dropout)

        multi_grids:add(grid)
    end

    return multi_grids
end

local function mgConvInput(nOutputPlanes, dropout)
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
        ConvBNReLU(proc, 3, nOutputPlanes[iG], 3, dropout)
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

function NET.createModel(opt)
    NET.packages()

    local model = nn.Sequential()

    local blocks = {
        {{40,40,40}, {3,3,3}},
        {{80,40,40}, {3,3,3}},
        {{160,80,40}, {3,3,3}},
        {{320,160,80}, {3,3,1}},
        {{320,240}, {3,1}},
    }

    local nIPs = {3,3,3}
    local nOPs
    for indBlock = 1,#blocks do
        nOPs = blocks[indBlock][1]
        local kernels = blocks[indBlock][2]
        for indLayer = 1,opt.nLayer do
            --local dropout = indLayer ~= opt.nLayer and 0.4 or nil
            local dropout = nil
            local multi_grids
            if indBlock == 1 and indLayer == 1 then
                multi_grids = mgConvInput(nOPs, dropout)
            else
                multi_grids = mgConv(nIPs, nOPs, kernels, dropout)
            end
            model:add(multi_grids)

            nIPs = {}
            for i, depth in ipairs(nOPs) do nIPs[i] = depth end

            if indLayer == opt.nLayer then
                local isConcat = kernels[#kernels] == 1 and true or false
                model:add(mgPool(nIPs, isConcat))
            end
        end
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

    return model
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
end

return NET
