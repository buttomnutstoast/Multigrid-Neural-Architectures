local NET = {}
function NET.packages()
    require 'cudnn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local Convolution = cudnn.SpatialConvolution
    local ReLU = cudnn.ReLU
    local Max = nn.SpatialMaxPooling
    local SBatchNorm = nn.SpatialBatchNormalization

    -- The shortcut layer is identity
    local function Shortcut(nInputPlane, nOutputPlane)
        if nInputPlane ~= nOutputPlane then
            -- zero-padded identity shortcut
            return nn.Padding(1, (nOutputPlane - nInputPlane), 3)
        else
            return nn.Identity()
        end
    end

    -- The basic residual layer block
    local function ResBlock(nIP, nOP, dropout)
        local s = nn.Sequential()
        if dropout and dropout > 0 then
            s:add(nn.Dropout(dropout))
        end
        s:add(Convolution(nIP,nOP,3,3,1,1,1,1))
        s:add(SBatchNorm(nOP))
        s:add(ReLU(true))
        if dropout and dropout > 0 then
            s:add(nn.Dropout(dropout))
        end
        s:add(Convolution(nOP,nOP,3,3,1,1,1,1))
        s:add(SBatchNorm(nOP))

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(Shortcut(nIP, nOP)))
            :add(nn.CAddTable(true))
            :add(ReLU(true))
    end

    local function ResBlockInput(nOP, dropout)
        local s = nn.Sequential()
        if dropout and dropout > 0 then
            s:add(nn.Dropout(dropout))
        end
        s:add(Convolution(nOP,nOP,3,3,1,1,1,1))
        s:add(SBatchNorm(nOP))
        s:add(ReLU(true))
        if dropout and dropout > 0 then
            s:add(nn.Dropout(dropout))
        end
        s:add(Convolution(nOP,nOP,3,3,1,1,1,1))
        s:add(SBatchNorm(nOP))

        return nn.Sequential()
            :add(Convolution(3,nOP,3,3,1,1,1,1))
            :add(SBatchNorm(nOP))
            :add(ReLU(true))
            :add(nn.ConcatTable()
                :add(s)
                :add(Shortcut(nOP, nOP)))
            :add(nn.CAddTable(true))
            :add(ReLU(true))
    end


    local blocks = {64,128,256,512,512}
    local dropouts = {nil,0.1,0.2,0.3,0.4}

    local model = nn.Sequential()
    local nIP = 3
    local nOP
    for indBlock = 1,#blocks do
        nOP = blocks[indBlock]
        for indLayer = 1,opt.nLayer do
            local dropout = opt.isDropout and dropouts[indBlock]
            local block
            if indBlock == 1 and indLayer == 1 then
                block = ResBlockInput(nOP, dropout)
            else
                block = ResBlock(nIP, nOP, dropout)
            end
            model:add(block)

            if indLayer == opt.nLayer then
                model:add(Max(2,2,2,2,0,0):ceil())
            end
            nIP = nOP
        end
    end

    local nLinear
    if opt.dataset == 'cifar10' then
        nLinear = 10
    else
        nLinear = 100
    end

    local classifier = nn.Sequential()
    classifier:add(nn.View(blocks[#blocks]))
    classifier:add(nn.Linear(blocks[#blocks], nLinear))
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
    cmd:option('-isDropout', false, 'if using dropout')
end

return NET
