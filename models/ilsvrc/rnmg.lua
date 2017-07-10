require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization
local UpSample = nn.SpatialUpSamplingNearest

-----------------
-- basic units --
-----------------
local function Shortcut(nIP, nOP)
    if nIP ~= nOP then
        -- zero-padded identity shortcut
        return nn.Padding(1, (nOP - nIP), 3)
    else
        return nn.Identity()
    end
end

local function ConvBNReLU(mod, nIP, nOP, kernel)
    local k = kernel
    local p = k == 1 and 0 or 1

    mod:add(Convolution(nIP, nOP, k, k, 1, 1, p, p))
    mod:add(SBatchNorm(nOP))
    mod:add(ReLU(true))
    return mod
end

local function ConvBN(mod, nIP, nOP, kernel)
    local k = kernel
    local p = k == 1 and 0 or 1

    mod:add(Convolution(nIP, nOP, k, k, 1, 1, p, p))
    mod:add(SBatchNorm(nOP))
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

local function mgConv(nInputPlanes, nOutputPlanes, kernels)
    -- Build a module consists of
    --    (<resample&concat>, <conv-BN,Relu>,
    --     <resample&concat>, <conv-BN>) + shortcut
    -- Args:
    --    nInputPlanes: dimension of input grids
    --    nOutputPlanes: dimension of output grids
    --    kernels: kernel size of each grid
    assert(#nInputPlanes == #nOutputPlanes,
        'number of input grid should be equal to output grid')
    assert(#nInputPlanes == #kernels,
        'should provide kernel size for every scale of grid')

    local mg_conv = nn.Sequential()

    local shortcut_convs = nn.ConcatTable()

    -- build convs blocks (<conv-BN-ReLU>, <conv-BN>)
    local convs = nn.Sequential()
    -- 1. <conv-BN-ReLU>
    local resample_concat, _nIPs = ResampleConcat(nInputPlanes)
    convs:add(resample_concat)
    local conv_bn_relu = nn.ParallelTable()
    for i = 1,#_nIPs do
        local mod = nn.Sequential()
        ConvBNReLU(mod, _nIPs[i], nOutputPlanes[i], kernels[i])
        conv_bn_relu:add(mod)
    end
    convs:add(conv_bn_relu)
    -- 3. <conv-BN>
    local resample_concat, _nIPs = ResampleConcat(nOutputPlanes)
    convs:add(resample_concat)
    local conv_bn = nn.ParallelTable()
    for i = 1,#_nIPs do
        local mod = nn.Sequential()
        ConvBN(mod, _nIPs[i], nOutputPlanes[i], kernels[i])
        conv_bn:add(mod)
    end
    convs:add(conv_bn)
    shortcut_convs:add(convs)

    -- build shortcuts
    local shortcut = nn.ParallelTable()
    for i = 1,#nInputPlanes do
        shortcut:add(Shortcut(nInputPlanes[i], nOutputPlanes[i]))
    end
    shortcut_convs:add(shortcut)

    -- add shortcut and convs
    local add_shortcut_convs = nn.ConcatTable()
    for i = 1,#nInputPlanes do
        local pick = nn.ConcatTable()
        local get_convs = nn.Sequential()
            :add(nn.SelectTable(1))
            :add(nn.SelectTable(i))
        local get_shortcut = nn.Sequential()
            :add(nn.SelectTable(2))
            :add(nn.SelectTable(i))
        pick:add(get_convs):add(get_shortcut)

        local sum = nn.Sequential()
        sum:add(pick):add(nn.CAddTable(true)):add(ReLU(true))
        add_shortcut_convs:add(sum)
    end

    mg_conv:add(shortcut_convs)
    mg_conv:add(add_shortcut_convs)
    return mg_conv
end

local function mgConvInput(nOutputPlanes)
    -- Build a module consists of
    --    <resample>, <conv-BN-ReLU>, <max-pool>
    -- Args:
    --    nOutputPlanes: dimension of output grids
    local mg_conv_input = nn.Sequential()

    -- resampling input image and <conv-BN-ReLU>
    local resample_image = nn.ConcatTable()
    for i = 1,#nOutputPlanes do
        local proc = nn.Sequential()
        if i == 1 then
            proc:add(nn.Identity())
        else
            local r = torch.pow(2, i-1)
            local resize = Avg(r, r, r, r, 0, 0)
            proc:add(resize)
        end
        local nOP = nOutputPlanes[i]
        proc:add(Convolution(3,nOP,7,7,2,2,3,3))
        proc:add(SBatchNorm(nOP))
        proc:add(ReLU(true))
        proc:add(Max(3,3,2,2,1,1))
        resample_image:add(proc)
    end
    mg_conv_input:add(resample_image)

    return mg_conv_input
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

local NET = {}
function NET.packages()
    require 'cudnn'
    -- set cudnn flag
    cudnn.fastest = true
    cudnn.benchmark = true
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local model = nn.Sequential()

    local inputBlock = {64,32,16} -- (224,112,56)->(56,28,14)
    model:add(mgConvInput(inputBlock))

    local cfg = {
        [18]  = {2, 2, 2, 2},
        [34]  = {3, 4, 6, 3},
    }

    local blocks = {
        -- {grid size, kernel size, isConcat}
        {{64,32,16}, {3,3,3}, false}, -- (56,28,14)->(28,14,7)
        {{128,64,32}, {3,3,3}, true}, -- (28,14,7)->(14,7)
        {{256,128}, {3,3}, true}, -- (14,7)->(7)
        {{512}, {3}, false}, -- (7,3)->3
    }

    local nIPs = inputBlock
    local nOPs
    for indBlock = 1,#blocks do
        nOPs = blocks[indBlock][1]
        local kernels = blocks[indBlock][2]
        local isConcat = blocks[indBlock][3]
        local nLayer = cfg[opt.depth][indBlock]

        for indLayer = 1,nLayer do
            local multi_grids
            multi_grids = mgConv(nIPs, nOPs, kernels)
            model:add(multi_grids)

            nIPs = {}
            for i, depth in ipairs(nOPs) do nIPs[i] = depth end

        end

        if indBlock < #blocks then
            model:add(mgPool(nIPs, isConcat))
        end
    end

    local classifier = nn.Sequential()
    classifier:add(nn.SelectTable(1))
    classifier:add(Avg(7, 7, 1, 1, 0, 0))
    classifier:add(nn.View(-1, nIPs[1]))
    classifier:add(nn.Linear(nIPs[1], 1000))
    classifier:add(nn.LogSoftMax())
    model:add(classifier)

    local function ConvInit(name)
        for k,v in pairs(model:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/n))
            v.bias:zero()
        end
    end
    local function BNInit(name)
        for k,v in pairs(model:findModules(name)) do
            v.weight:fill(1)
            v.bias:zero()
        end
    end

    ConvInit('cudnn.SpatialConvolution')
    ConvInit('nn.SpatialConvolution')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')
    for k,v in pairs(model:findModules('nn.Linear')) do
        v.bias:zero()
    end

    if opt.cudnn == 'deterministic' then
        model:apply(function(m)
            if m.setMode then m:setMode(1,1,1) end
        end)
    end

    model:get(1).gradInput = nil

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
    info[#info+1] = utilfuncs.newInfoEntry('top5',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err, iterSize)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    assert(batch_size == labelsCPU:size(1))

    info[1].value   = err * iterSize
    info[1].N       = batch_size

    info[2].value   = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N       = batch_size

    info[3].value   = mathfuncs.topK(outputsCPU, labelsCPU, 5)
    info[3].N       = batch_size
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top5',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size

    info[2].value = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N     = batch_size

    info[3].value = mathfuncs.topK(outputsCPU, labelsCPU, 5)
    info[3].N     = batch_size
end

function NET.trainRule(currentEpoch, opt)
    -- learning decay 0.2 every 60 epochs
    local start = 1e-1 -- LR: 10^-(star) ~ 10^-(start)*decay^sum
    local decay = 0.1
    local lr = start * torch.pow(decay, math.floor((currentEpoch-1)/30))
    return {LR= lr, WD= 1e-4}
end

function NET.arguments(cmd)
    cmd:option('-depth', 34, 'number of layers per block')
end

return NET
