utilfuncs = {}

function utilfuncs.recursivePut2Gpu(Atable, AgpuTable)
    for i=1,#Atable do
        if torch.type(Atable[i]) == 'table' then
            if AgpuTable[i] == nil then
                AgpuTable[i] = {}
            end
            utilfuncs.recursivePut2Gpu(Atable[i], AgpuTable[i])
        else
            if AgpuTable[i] == nil then
                AgpuTable[i] = torch.CudaTensor()
            end
            AgpuTable[i]:resize(Atable[i]:size()):copy(Atable[i])
        end
    end
end

function utilfuncs.put2GPU(cpuData, gpuLocation)
    local dataInd = 0
    if torch.type(gpuLocation) == 'table' then
        utilfuncs.recursivePut2Gpu(cpuData, gpuLocation)
    else
        if #cpuData == 1 then
            gpuLocation:resize(cpuData[1]:size()):copy(cpuData[1])
        else
            error('Some kind of error...')
        end
    end
end

function utilfuncs.newInfoEntry(vname, vval, vn, store_all)
    local storeAll = store_all or false -- if true, vn will be ignored
    return {name=vname, value=vval, N=vn, store=storeAll}
end

return utilfuncs
