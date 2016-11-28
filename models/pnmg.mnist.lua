local NET = {}
function NET.packages()
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
local function ConvBNReLU(mod, nIP, nOP)
    mod:add(Convolution(nIP, nOP, 3, 3, 1, 1, 1, 1))
    mod:add(SBatchNorm(nOP, 1e-3))
    mod:add(ReLU(true))
    return mod
end

local function ConvBN(mod, nIP, nOP)
    mod:add(Convolution(nIP, nOP, 3, 3, 1, 1, 1, 1))
    mod:add(SBatchNorm(nOP, 1e-3))
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

local function mgConv(nInputPlanes, nOutputPlanes, isDrop)
    -- Build a module which take multi-scale grids as input and output
    -- another multi-scale grids:
    --  inputGrids: {input_scale_0, input+scale_1, ..., input_scale_n}
    --  outputGrids: {output_scale_0, outptu_scale_1, ..., output_scale_n}
    -- Args:
    --  nInputPlanes: dimension of input grids
    --  nOutputPlanes: dimension of output grids
    local mg_conv = nn.Sequential()
    local resample_concat, _nIPs = ResampleConcat(nInputPlanes, isDrop)
    mg_conv:add(resample_concat)

    local convs = nn.ParallelTable()
    for i = 1,#_nIPs do
        local _conv = nn.Sequential()
        ConvBNReLU(_conv, _nIPs[i], nOutputPlanes[i])
        convs:add(_conv)
    end
    mg_conv:add(convs)

    return mg_conv
end

local function mgConvOutput(nInputPlanes, nOutputPlanes, isDrop)
    -- same as mgConv, but use ConvBN instead of ConvBNReLU
    local mg_conv = nn.Sequential()
    local resample_concat, _nIPs = ResampleConcat(nInputPlanes, isDrop)
    mg_conv:add(resample_concat)

    local convs = nn.ParallelTable()
    for i = 1,#_nIPs do
        local _conv = nn.Sequential()
        ConvBN(_conv, _nIPs[i], nOutputPlanes[i])
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
        ConvBNReLU(proc, 1, nOutputPlanes[iG])
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
                for j = #nOPs-nGrid+1, #nOPs do
                    _select:add(nn.SelectTable(j))
                    _nOPs[#_nOPs+1] = nOPs[j]
                end
                _mg_conv:add(_select)

                _mg_conv:add(mgConv(_nOPs, _nOPs))
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
                ConvBNReLU(_conv, nOPs[#nOPs], nOPs[#nOPs])
                convs:add(_conv)
                model:add(convs)
            end
        end
    end
end

local function MultiGrids(model, nIPs, nOPs, nLayer, isDrop)
    -- <mg-conv> x nLayer
    for i = 1,nLayer do
        local _drop = i == 1 and isDrop or false
        model:add(mgConv(nIPs, nOPs, _drop))
        update(nIPs, nOPs)
    end
end

local function MultiGridsOutput(model, nIPs, nOPs, nLayer, isDrop)
    -- <mg-conv> x nLayer
    for i = 1,nLayer do
        local _drop = i == 1 and isDrop or false
        local _mgconv = i == nLayer and mgConvOutput or mgConv
        model:add(_mgconv(nIPs, nOPs, _drop))
        update(nIPs, nOPs)
    end
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
    -- building multi-grids
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
