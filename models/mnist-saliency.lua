local NET = {}
function NET.packages()
    require 'cudnn'
    if not nn.ConcatUnet then dofile('layers/ConcatUnet.lua') end
end

function NET.createModel(opt)
    NET.packages()
    local model = torch.load(opt.trainedNet)
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

function NET.arguments(cmd)
    cmd:option('-trainedNet', '/path/to/trained/net', 'path to trained net')
end

return NET

