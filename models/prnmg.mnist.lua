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
    if nOP > nIP then
        -- zero-padded identity shortcut
        return nn.Padding(1, (nOP - nIP), 3)
    elseif nIP > nOP then
        local conv = nn.Sequential()
        conv:add(Convolution(nIP, nOP, 1, 1, 1, 1, 0, 0))
        conv:add(SBatchNorm(nOP))
        return conv
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

local function ResampleConcat(nIPs, isDrop)
    -- helper function to output module which perform
    -- resampling, followed by concatenation
    local resample_concat = nn.ConcatTable()
    local nOPs = {}

    local nGrids = isDrop and #nIPs-1 or #nIPs
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

local function resConv(nIP, nOP, kernel)
    -- <conv-BN-Relu>,<conv-BN> + shortcut
    local s = nn.Sequential()
    ConvBNReLU(s, nIP, nOP, kernel)
    ConvBN(s, nOP, nOP, kernel)

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(Shortcut(nIP, nOP)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
end

local function mgConv(nInputPlanes, nOutputPlanes, kernel, isDrop, isOut)
    -- Build a module consists of
    --    (<BN>, <resample&concat>, <conv-BN,Relu>,
    --     <resample&concat>, <conv-BN>) + shortcut
    -- Args:
    --    nInputPlanes: dimension of input grids
    --    nOutputPlanes: dimension of output grids
    local mg_conv = nn.Sequential()

    local shortcut_convs = nn.ConcatTable()

    -- build convs blocks (<conv-BN-ReLU>, <conv-BN>)
    local convs = nn.Sequential()
    -- 1. <conv-BN-ReLU>
    local resample_concat, _nIPs = ResampleConcat(nInputPlanes, isDrop)
    convs:add(resample_concat)
    local conv_bn_relu = nn.ParallelTable()
    for i = 1,#_nIPs do
        local mod = nn.Sequential()
        ConvBNReLU(mod, _nIPs[i], nOutputPlanes[i], kernel)
        conv_bn_relu:add(mod)
    end
    convs:add(conv_bn_relu)
    -- 3. <conv-BN>
    local resample_concat, _nIPs = ResampleConcat(nOutputPlanes, false)
    convs:add(resample_concat)
    local conv_bn = nn.ParallelTable()
    for i = 1,#_nIPs do
        local mod = nn.Sequential()
        ConvBN(mod, _nIPs[i], nOutputPlanes[i], kernel)
        conv_bn:add(mod)
    end
    convs:add(conv_bn)
    shortcut_convs:add(convs)

    -- build shortcuts
    local shortcut = nn.ConcatTable()
    local nShortcut = #_nIPs
    for i = 1,nShortcut do
        local _sc = nn.Sequential()
        _sc:add(nn.SelectTable(i))
        _sc:add(Shortcut(nInputPlanes[i], nOutputPlanes[i]))
        shortcut:add(_sc)
    end
    shortcut_convs:add(shortcut)

    -- add shortcut and convs
    local add_shortcut_convs = nn.ConcatTable()
    for i = 1,nShortcut do
        local pick = nn.ConcatTable()
        local get_convs = nn.Sequential()
            :add(nn.SelectTable(1))
            :add(nn.SelectTable(i))
        local get_shortcut = nn.Sequential()
            :add(nn.SelectTable(2))
            :add(nn.SelectTable(i))
        pick:add(get_convs):add(get_shortcut)

        local sum = nn.Sequential()
        sum:add(pick):add(nn.CAddTable(true))
        if not isOut then sum:add(ReLU(true)) end
        add_shortcut_convs:add(sum)
    end

    mg_conv:add(shortcut_convs)
    mg_conv:add(add_shortcut_convs)
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
        ConvBNReLU(proc, 1, nOutputPlanes[iG], 3)
        mg_inputs:add(proc)
    end
    return mg_inputs
end

----------------------
-- helper functions --
----------------------
local function update(old, new)
    local nTb = #old > #new and #old or #new
    for i = 1,nTb do old[i] = new[i] end
end

local function MultiGridsInput(model, nIPs, nOPs, nLayer)
    -- always perform
    -- <mg-conv-input> x 1
    -- <res-conv> x nLayer (3rd grid)
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
                for j = #nOPs-nGrid+1, #nOPs do
                    _select:add(nn.SelectTable(j))
                    _nOPs[#_nOPs+1] = nOPs[j]
                end
                _mg_conv:add(_select)

                _mg_conv:add(mgConv(_nOPs, _nOPs, 3, false, false))
                mg_convs:add(_mg_conv)
                model:add(mg_convs)
                model:add(nn.FlattenTable())
            end
        else
            -- <res-conv>
            for i = 1,nLayer do
                local convs = nn.ParallelTable()
                for j = 1,#nOPs-1 do
                    convs:add(nn.Identity())
                end

                convs:add(resConv(nOPs[#nOPs], nOPs[#nOPs], 3))
                model:add(convs)
            end
        end
    end
end

local function MultiGrids(model, nIPs, nOPs, nLayer, isDrop)
    -- <mg-conv> x nLayer
    for i = 1,nLayer do
        local _drop = i == 1 and isDrop or false
        model:add(mgConv(nIPs, nOPs, 3, _drop, false))
        update(nIPs, nOPs)
    end
end

local function MultiGridsOutput(model, nIPs, nOPs, nLayer, isDrop)
    -- <mg-conv> x nLayer
    for i = 1,nLayer do
        local _drop = i == 1 and isDrop or false
        local _kernel = i == nLayer and 1 or 3
        local _isOut = i == nLayer and true or false
        model:add(mgConv(nIPs, nOPs, _kernel, _drop, _isOut))
        update(nIPs, nOPs)
    end
end

local NET = {}
function NET.packages()
    require 'cudnn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local model = nn.Sequential()

    local nClass = opt.dataset == 'mnist-seg' and 10 or 1
    local blocks = {
        {{64,32,16,8}, false},
        {{64,32,16,8}, false},
        {{64,32,16,8}, false},
        {{64,32,16,8}, false},
        {{64,32,16}, true},
        {{64,32}, true},
        {{nClass}, true}
    }

    local nIPs = {1,1,1,1}
    local nOPs
    for indBlock = 1,#blocks do
        nOPs = blocks[indBlock][1]
        local isDrop = blocks[indBlock][2]

        if indBlock == 1 then
            MultiGridsInput(model, nIPs, nOPs, opt.nLayer)
        elseif indBlock == #blocks then
            MultiGridsOutput(model, nIPs, nOPs, opt.nLayer, isDrop)
        else
            MultiGrids(model, nIPs, nOPs, opt.nLayer, isDrop)
        end
        update(nIPs, nOPs)
    end

    model:add(nn.SelectTable(1))
    model:add(nn.Sigmoid())

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
    criterion:add(nn.BCECriterion())
    return criterion
end

function NET.trainOutputInit()
    local info = {}
    -- utilfuncs.newInfoEntry is defined in utils/train_eval_test_func.lua
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err, iterSize)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    assert(batch_size == labelsCPU:size(1))

    info[1].value   = err * iterSize
    info[1].N       = batch_size
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size
end

function NET.trainRule(currentEpoch, opt)
    -- learning decay 0.2 every 60 epochs
    local decay_epoch = {60,120,160}
    local sum = 0
    for i = 1,#decay_epoch do
        if currentEpoch >= decay_epoch[i] then
            sum = sum + 1
        end
    end
    local start = 1e-1 -- LR: 10^-(star) ~ 10^-(start)*decay^sum
    local decay = 0.2
    local lr = start * torch.pow(decay, sum)
    return {LR= lr, WD= 5e-4}
end

function NET.arguments(cmd)
    cmd:option('-nLayer', 1, 'number of layers per block')
end

return NET
