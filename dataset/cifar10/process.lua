function getDataLabel(rootDir, protocol)
    local protocol = protocol or 'train'

    local data = torch.FloatTensor()
    local labels = torch.FloatTensor()

    if protocol == 'train' then
        data:resize(3072, 50000)
        labels:resize(50000)
        for i = 1,5 do
            local dat_path = paths.concat(rootDir, 'data_batch_' .. i .. '.t7')
            local dat = torch.load(dat_path, 'ascii')
            local start = (i-1) * 10000
            data[{{}, {start+1, start+10000}}]:copy(dat.data)
            labels[{{start+1, start+10000}}]:copy(dat.labels)
        end
    elseif protocol == 'test' then
        data:resize(3072, 10000)
        labels:resize(10000)
        local dat_path = paths.concat(rootDir, 'test_batch.t7')
        local dat = torch.load(dat_path, 'ascii')
        data:copy(dat.data)
        labels:copy(dat.labels)
    else
        error('undefined protocol ' .. protocol)
    end

    return data, labels
end
