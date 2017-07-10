local NET = {}
function NET.packages()
    require 'cudnn'
    require 'cunn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
    if not nn.ConcatUnet then dofile('layers/ConcatUnet.lua') end
end

require 'cudnn'
local Convolution = cudnn.SpatialConvolution
local UpConvolution = cudnn.SpatialFullConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local SBatchNorm = nn.SpatialBatchNormalization
local UpSample = nn.SpatialUpSamplingNearest

-----------------
-- basic units --
-----------------
local function ConvBNReLU(mod, nIP, nOP)
    mod:add(Convolution(nIP, nOP, 3, 3, 1, 1, 1, 1))
    mod:add(SBatchNorm(nOP, 1e-3))
    mod:add(ReLU(true))
    return mod
end

local function ConvBN(mod, nIP, nOP)
    mod:add(Convolution(nIP, nOP, 1, 1, 1, 1, 0, 0))
    mod:add(SBatchNorm(nOP, 1e-3))
    return mod
end

local function UpConvBNReLU(mod, nIP, nOP)
    mod:add(UpConvolution(nIP, nOP, 2, 2, 2, 2, 0, 0))
    mod:add(SBatchNorm(nOP, 1e-3))
    mod:add(ReLU(true))
    return mod
end

local function mgUpConv(nInputPlanes, nOutputPlanes)
    assert(#nInputPlanes == #nOutputPlanes,
        'number of input grid should be equal to output grid')
    local upconvs = nn.ParallelTable()
    for i = 1,#nInputPlanes do
        local mod = nn.Sequential()
        UpConvBNReLU(mod, nInputPlanes[i], nOutputPlanes[i])
        upconvs:add(mod)
    end
    return upconvs
end

local function mgConv(nInputPlanes, nOutputPlanes, isReLU)
    -- Build a module which take multi-scale grids as input and output another
    -- multi-scale grids:
    --    inputGrids: {input_scale_0, input+scale_1, ..., input_scale_n}
    --    outputGrids: {output_scale_0, outptu_scale_1, ..., output_scale_n}
    -- Args:
    --    nInputPlanes: dimension of input grids
    --    nOutputPlanes: dimension of output grids
    assert(#nInputPlanes == #nOutputPlanes,
        'number of input grid should be equal to output grid')

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
        grid:add(nn.JoinTable(2))
        if isReLU then
            ConvBNReLU(grid, nInput, nOutputPlanes[iG])
        else
            ConvBN(grid, nInput, nOutputPlanes[iG])
        end

        multi_grids:add(grid)
    end

    return multi_grids
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

local function mgPool(nInputPlanes, isDrop)
    local mg_pool = nn.ConcatTable()
    -- max-pool or concatenate
    local nGrids = #nInputPlanes
    for i = 1,nGrids do
        if i == nGrids and isDrop then
            -- update nInputPlanes
            nInputPlanes[i] = nil
        else
            local proc = nn.Sequential()
            proc:add(nn.SelectTable(i))
            proc:add(Max(2,2,2,2,0,0):ceil())
            mg_pool:add(proc)
        end
    end

    return mg_pool
end

---------------------
-- helper function --
---------------------
local function caddtable(tab1, tab2)
    local nTab = #tab1 > #tab2 and #tab1 or #tab2
    local sum = {}
    for i = 1,nTab do
        local val1 = tab1[i] or 0
        local val2 = tab2[i] or 0
        sum[i] = val1 + val2
    end
    return sum
end

local function deepcopy(tab)
    local new = {}
    for key, val in pairs(tab) do
        if torch.type(val) == 'table' then
            new[key] = deepcopy(val)
        else
            new[key] = val
        end
    end
    return new
end

function NET.createModel(opt)
    NET.packages()

    local model = nn.Sequential()

    local blocks = {
        {{64,32,16}, false},
        {{128,64,32}, true},
        {{256,128}, true},
        {{512}, nil}
    }

    local nIP = 1
    local function Unet(depth)
        local unetIP = nIP
        local unetOP = blocks[depth][1]
        local isDrop = blocks[depth][2]
        local model = nn.Sequential()

        if depth == #blocks then
            -- conv and upconv
            model:add(mgConv(unetIP, unetOP, true))
            model:add(mgUpConv(unetOP, unetIP))
        else
            if depth > 1 then
                model:add(mgConv(unetIP, unetOP, true))
            else
                model:add(mgConvInput(unetOP))
            end
            nIP = deepcopy(unetOP)

            -- concat with sub unet
            local shortcut_subnet = nn.ConcatTable()
            local mg_pool = mgPool(nIP, isDrop)
            local subnet, subnetOP = Unet(depth+1) -- update nIP
            local shortcut = nn.Identity()
            local pool_subnet = nn.Sequential()
                :add(mg_pool)
                :add(subnet)
            shortcut_subnet:add(shortcut)
            shortcut_subnet:add(pool_subnet)

            model:add(shortcut_subnet)
            model:add(nn.ConcatUnet())
            model:add(nn.MapTable(nn.JoinTable(2)))
            local sumOP = caddtable(unetOP, subnetOP)

            -- conv and upconv
            model:add(mgConv(sumOP, unetOP, true))
            if depth > 1 then
                model:add(mgUpConv(unetOP, unetIP))
            else
                local nOP = opt.dataset == 'mnist-seg' and 10 or 1
                model:add(mgConv(unetOP, {nOP,nOP,nOP}, false))
                model:add(nn.SelectTable(1))
            end
        end
        return model, unetIP
    end

    local model = Unet(1)
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

    return model
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
    info[1].value   = err * iterSize
    info[1].N       = batch_size
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('meanIU',0,0)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size

    info[2].value = mathfuncs.mnistIU(outputs, labelsCPU, 0.5)
    info[2].N = batch_size
end

function NET.trainRule(currentEpoch, opt)
    local delta = 3
    local start = 1 -- LR: 10^-(star) ~ 10^-(start + delta)
    local ExpectedTotalEpoch = opt.nEpochs
    return {LR= 10^-((currentEpoch-1)*delta/(ExpectedTotalEpoch-1)+start),
           WD= 5e-4}
end

function NET.arguments(cmd)
end

return NET
