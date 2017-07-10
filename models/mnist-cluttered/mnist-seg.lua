local NET = {}
function NET.packages()
    require 'cudnn'
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
    if not nn.ConcatUnet then dofile('layers/ConcatUnet.lua') end
end

function NET.createModel(opt)
    NET.packages()
end

function NET.createCriterion()
    local criterion = nn.MultiCriterion()
    criterion:add(nn.BCECriterion())
    return criterion
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('meanIU',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('prediction',0,0, true)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size

    info[2].value = mathfuncs.mnistIU(outputs, labelsCPU, 0.5)
    info[2].N = batch_size

    info[3].value = outputs:float()
end

return NET
