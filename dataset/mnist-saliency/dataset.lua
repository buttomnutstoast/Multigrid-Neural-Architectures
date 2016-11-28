require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'xlua'


local dataset = torch.class('dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
        A dataset class for mnist-spatial transformation dataset
    ]],
    {name="path",
     type="string",
     help="Path to root directory of mnist-spatial transformation"},

    {name="protocol",
     type="string",
     default="train",
     help="train | test"}
}

function dataset:__init(...)

    -- argcheck
    local args =  initcheck(...)
    print(args)
    for k,v in pairs(args) do self[k] = v end

    self.sampleHookTrain = self.defaultSampleHook
    self.sampleHookTest = self.defaultSampleHook

    -- get dataset
    local filePath = paths.concat(self.path, 'mnist_translation_3.t7')
    local file = torch.load(filePath)[self.protocol]

    self.imageData = file.data
    self.imageClass = file.labels

    -- occlusion mask
    local mask = {}
    local stride = 2
    local size = 8
    local y = 1
    while y+size-1 <= 64 do
        local x = 1
        while x+size-1 <= 64 do
            local _mask = torch.Tensor(1,1,64,64):zero()
            _mask[{{}, {}, {x, x+size-1}, {y, y+size-1}}] = 1
            table.insert(mask, _mask)
            x = x + stride
        end
        y = y + stride
    end
    self.mask = torch.cat(mask, 1)
end

-- default image hooker
function dataset:defaultSampleHook(img)
    return img
end

-- size(), size(class)
function dataset:size()
    return self.imageData:size(1)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, tab)
    local tensor
    local quantity = #tab
    local iSize = torch.isTensor(tab[1]) and tab[1]:size():totable() or {}
    local tSize = {quantity}
    for _, dim in ipairs(iSize) do table.insert(tSize, dim) end
    tensor = torch.Tensor(table.unpack(tSize)):fill(-1)
    for i=1,quantity do
        tensor[i] = tab[i]
    end
    return tensor
end

function dataset:get(i1, i2)
    local indices = torch.range(i1, i2);
    local quantity = i2 - i1 + 1;
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        -- load the sample
        local img = self.imageData[indices[i]]
        local out = self:sampleHookTest(img)
        table.insert(dataTable, out)
        table.insert(scalarTable, self.imageClass[indices[i]])
    end
    local data = tableToOutput(self, dataTable):squeeze(1)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:getInputs(i1, i2, currentEpoch)
    local data, scalarLabels = self:get(i1, i2)
    return {data}, {scalarLabels}
end

return dataset
