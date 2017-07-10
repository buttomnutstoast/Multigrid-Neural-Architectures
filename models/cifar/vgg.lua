local NET = {}
function NET.packages()
    require 'cudnn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local Convolution = cudnn.SpatialConvolution
    local ReLU = nn.ReLU
    local Max = nn.SpatialMaxPooling
    local SBatchNorm = nn.SpatialBatchNormalization

    local vgg = nn.Sequential()

    -- building block
    local function ConvBNReLU(nInputPlane, nOutputPlane, dropout)
        vgg:add(Convolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
        vgg:add(SBatchNorm(nOutputPlane, 1e-3))
        vgg:add(ReLU(true))
        if dropout and dropout > 0 then
            vgg:add(nn.Dropout(dropout))
        end
        return vgg
    end

    local blocks = {102,204,408,816,816}

    local nIP = 3
    local nOP
    for indBlock = 1,#blocks do
        nOP = blocks[indBlock]
        for indLayer = 1,opt.nLayer do
            if indLayer == opt.nLayer then
                ConvBNReLU(nIP, nOP)
                vgg:add(Max(2,2,2,2):ceil())
            else
                --ConvBNReLU(nIP, nOP, 0.4)
                ConvBNReLU(nIP, nOP)
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
    classifier:add(nn.Linear(816,nLinear))
    classifier:add(nn.LogSoftMax())
    vgg:add(classifier)

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

    MSRinit(vgg)

    return vgg
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
    -- exponentially decay
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
