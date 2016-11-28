local mnist_cluttered = require 'mnist_cluttered'

local config = {
  task = 'centered_transform',
  megapatch_w = 64,
  nDigits = 1,
  border = 0,
  scale = {1,1},
  angle = 60,
  seed = 1,
  threshold = 0.3
}
config.datasetPath = 'mnist/train.t7'
trainDataInfo = mnist_cluttered.createData(config)

local nTrain = 50000
local trainData = torch.FloatTensor(nTrain, 1, config.megapatch_w, config.megapatch_w)
local trainLabel = torch.FloatTensor(nTrain, 1, config.megapatch_w, config.megapatch_w)
for i = 1,nTrain do
  local observation, target = unpack(trainDataInfo.nextExample())
  trainData[i] = observation
  trainLabel[i] = target
end

config.datasetPath = 'mnist/valid.t7'
valDataInfo = mnist_cluttered.createData(config)

local nVal = 10000
local valData = torch.FloatTensor(nVal, 1, config.megapatch_w, config.megapatch_w)
local valLabel = torch.FloatTensor(nVal, 1, config.megapatch_w, config.megapatch_w)
for i = 1,nVal do
  local observation, target = unpack(valDataInfo.nextExample())
  valData[i] = observation
  valLabel[i] = target
end

config.datasetPath = 'mnist/test.t7'
testDataInfo = mnist_cluttered.createData(config)

local nTest = 10000
local testData = torch.FloatTensor(nTest, 1, config.megapatch_w, config.megapatch_w)
local testLabel = torch.FloatTensor(nTest, 1, config.megapatch_w, config.megapatch_w)
for i = 1,nTest do
  local observation, target = unpack(testDataInfo.nextExample())
  testData[i] = observation
  testLabel[i] = target
end

local results = {}
results.train = {data=torch.cat({trainData,valData},1), labels=torch.cat({trainLabel,valLabel},1)}
results.test = {data=testData, labels=testLabel}
torch.save('mnist_rotation.t7', results)
