require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'


local dataset = torch.class('dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
        A dataset class for images in a flat folder structure (folder-name is
        class-name). Optimized for extremely large datasets (upwards of 14
        million images). Tested only on Linux (as it uses command-line linux
        utilities to scale up)
    ]],
    {name="path",
     type="string",
     help="Path to root directory of Cifar10"},

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
    local filePath = paths.concat(self.path, 'cifar100_whitened.t7')
    local file = torch.load(filePath)

    local fileData
    if self.protocol == 'train' then
        fileData = file.trainData
    elseif self.protocol == 'test' then
        fileData = file.testData
    else
        error('Unidentified protocol:' .. self.protocol)
    end

    self.imageData = fileData.data:float():view(-1, 3072)
    self.imageClass = fileData.labels:float()

    self.image_size = self.imageData:size(1)

    self.classes = {}
    for i = 1,100 do table.insert(self.classes, i) end

    self.classList = {}
    for i=1,self.imageData:size(1) do
        local lab = self.imageClass[i]
        self.classList[lab] = self.classList[lab] or {}
        local len = #self.classList[lab]
        self.classList[lab][len+1] = i
    end

    self.classListSample = {}
    for i=1,#self.classes do
        self.classListSample[i] = torch.LongTensor(self.classList[i])
    end
end

-- default image hooker
function dataset:defaultSampleHook(img)
    local input = img:view(3,32,32)
    return input
end

-- size(), size(class)
function dataset:size()
    return self.image_size
end

-- getByClass
function dataset:getByClass(class)
    local index = math.max(1,
        math.ceil(torch.uniform() * self.classListSample[class]:nElement()))
    local class_ind = self.classListSample[class][index]
    local img = self.imageData[class_ind]
    return self:sampleHookTrain(img)
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

-- sampler, samples from the training set.
function dataset:sample(quantity)
    assert(quantity)
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        local class = torch.random(1, #self.classes)
        local out = self:getByClass(class)
        table.insert(dataTable, out)
        table.insert(scalarTable, class)
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:genInputs(quantity, currentEpoch)
    local data, scalarLabels = self:sample(quantity)
    return {data}, {scalarLabels}
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
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:getInputs(i1, i2, currentEpoch)
    local data, scalarLabels = self:get(i1, i2)
    return {data}, {scalarLabels}
end

return dataset
