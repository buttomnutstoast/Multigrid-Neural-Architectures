require 'cunn'

local function isContainer(module)
    -- only accept standard containers
    local moduleType = torch.type(module)
    local containers = {
        'nn.Sequential',
        'nn.Concat',
        'nn.DepthConcat',
        'nn.Parallel',
        'nn.ConcatTable',
        'nn.ParallelTable',
        }
    for i = 1,#containers do
        if moduleType == containers[i] then return true end
    end
    return false
end

local function copyContainer(module)
    local modType = torch.type(module)
    modType = modType:sub(4, #modType)
    newModel = nn[modType]
    if modType == 'Concat' or modType == 'DepthConcat' then
        return newModel(module.dimension)
    elseif modType == 'Parallel' then
        return newModel(module.inputDimension, module.outputDimension)
    else
        return newModel()
    end
end

local function cleanDPT(module)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newModel
    if torch.isTypeOf(module, nn.DataParallelTable) then
        newModel = nn.DataParallelTable(1, true, true)
        cutorch.setDevice(OPT.GPU)
        newModel:add(module:get(1), OPT.GPU)
    elseif isContainer(module) then
        newModel = copyContainer(module)
        for _, mod in ipairs(module.modules) do
            newModel:add(cleanDPT(mod))
        end
    else
        newModel = module
    end
    return newModel
end

local function retrieveDPT(module, nGPU, net)
    -- This helps to search for DPT which deeply lies in the nn.Container, and
    -- copy to different GPUs
    if torch.isTypeOf(module, nn.Container) then
        for i, mod in ipairs(module.modules) do
            if torch.type(mod) == 'nn.DataParallelTable' then
                module[i] = makeDataParallel(mod:get(1), nGPU, net)
            else
                retrieveDPT(mod, nGPU, net)
            end
        end
    end
end

local function removeDPT(module)
    -- This helps to replace nn.DataParallelTable with nn.Sequential which
    -- deeply lies in nn.Container
    if torch.isTypeOf(module, nn.Container) then
        for i, mod in ipairs(module.modules) do
            if torch.type(mod) == 'nn.DataParallelTable' then
                module[i] = mod:get(1):clone():cuda()
            else
                removeDPT(mod)
            end
        end
    end
end

function makeDataParallel(model, nGPU, net)
    -- This function clones the specified model from major GPU to other GPUs
    if nGPU >= 1 then
        print('converting module to nn.DataParallelTable')
        assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
        local model_single = model
        model = nn.DataParallelTable(1,true,true)
        for i=1, nGPU do
            cutorch.setDevice(i)
            model:add(model_single:clone():cuda(), i)
        end

        -- allow multi-threads for multi-GPUS
        local netobj = net
        local initFun = function()
            netobj.packages()
        end
        model:threads(initFun)
    end

    cutorch.setDevice(OPT.GPU)
    return model
end

function saveDataParallel(filename, model)
    --[[
        snapshotting models
    --]]
    -- clear the intermediate states in the model before saving to disk
    -- this saves lots of disk space
    local mods = model:listModules()
    for modInd=1,#mods do
        if mods[modInd].output then
            if torch.isTensor(mods[modInd].output) then
                mods[modInd].output = mods[modInd].output.new()
            elseif type(mods[modInd].output) == 'table' then
                mods[modInd].output = {}
            else
                mods[modInd].output = nil
            end
        end
        if mods[modInd].gradInput then
            if torch.isTensor(mods[modInd].gradInput) then
                mods[modInd].gradInput = mods[modInd].gradInput.new()
            elseif type(mods[modInd].gradInput) == 'table' then
                mods[modInd].gradInput = {}
            else
                mods[modInd].gradInput = nil
            end
        end
    end

    local tmpModel = cleanDPT(model)
    torch.save(filename, tmpModel)
end

function loadDataParallel(filename, nGPU, net)
    -- load require packages
    net.packages()

    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1), nGPU, net)
    else
        retrieveDPT(model, nGPU, net)
        return model
    end
end

function loadAndRemoveDPT(filename, net)
    net.packages()

    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
        return model:get(1):clone():cuda()
    else
        removeDPT(model)
        return model
    end
end
