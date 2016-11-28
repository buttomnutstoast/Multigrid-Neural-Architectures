local NET = {}
function NET.packages()
    require 'cudnn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local Convolution = cudnn.SpatialConvolution
    local UpConvolution = cudnn.SpatialFullConvolution
    local ReLU = nn.ReLU
    local SBatchNorm = nn.SpatialBatchNormalization
    local Max = nn.SpatialMaxPooling

    -- building block
    local function ConvBlock(mod, nIP, nOP)
        mod:add(Convolution(nIP, nOP, 3,3, 1,1, 1,1))
        mod:add(SBatchNorm(nOP, 1e-3))
        mod:add(ReLU(true))
    end

    local function UpConvBlock(mod, nIP, nOP)
        mod:add(UpConvolution(nIP, nOP, 2,2, 2,2, 0,0))
        mod:add(SBatchNorm(nOP, 1e-3))
        mod:add(ReLU(true))
    end

    local blocks = {64,128,256,512}

    local nIP = 1
    local function Unet(depth)
        local unetIP = nIP
        local unetOP = blocks[depth]
        local model = nn.Sequential()

        if depth == #blocks then
            -- conv and upconv
            ConvBlock(model, unetIP, unetOP)
            UpConvBlock(model, unetOP, unetIP)
        else
            ConvBlock(model, unetIP, unetOP)
            nIP = unetOP

            -- concat with sub unet
            local shortcut_subnet = nn.ConcatTable()
            local subnet, subnetOP = Unet(depth+1)
            local shortcut = nn.Identity()
            local pool_subnet = nn.Sequential()
                :add(Max(2,2,2,2,0,0))
                :add(subnet)
            shortcut_subnet:add(shortcut)
            shortcut_subnet:add(pool_subnet)

            model:add(shortcut_subnet)
            model:add(nn.JoinTable(2))

            -- conv and upconv
            ConvBlock(model, unetOP+subnetOP, unetOP)
            if depth > 1 then
                UpConvBlock(model, unetOP, unetIP)
            else
                local nOP = opt.dataset == 'mnist-seg' and 10 or 1
                model:add(Convolution(unetOP, nOP, 1,1, 1,1, 0,0))
                model:add(SBatchNorm(nOP, 1e-3))
            end
        end
        return model, unetIP
    end

    local unet = Unet(1)
    unet:add(nn.Sigmoid())

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

    MSRinit(unet)

    return unet
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
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size
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
end

return NET

