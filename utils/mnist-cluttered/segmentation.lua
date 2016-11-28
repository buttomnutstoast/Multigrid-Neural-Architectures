local mnist_cluttered = require 'mnist_cluttered'

local trainDataConfig = {
  datasetPath = 'mnist/train.t7',
  megapatch_w=64,
  nDigits=5,
  border=0,
  scale={0.5,1.25},
  angle=60,
  seed=1,
  threshold=0.3
}
trainDataInfo = mnist_cluttered.createData(trainDataConfig)

local nTrain = 10000
local trainData = torch.FloatTensor(nTrain, 1, trainDataConfig.megapatch_w, trainDataConfig.megapatch_w)
local trainLabel = torch.FloatTensor(nTrain, 10, trainDataConfig.megapatch_w, trainDataConfig.megapatch_w)
for i = 1,nTrain do
  local observation, target = unpack(trainDataInfo.nextExample())
  trainData[i] = observation
  trainLabel[i] = target
end

local valDataConfig = {
  datasetPath = 'mnist/valid.t7',
  megapatch_w=64,
  nDigits=5,
  border=0,
  scale={0.5,1.25},
  angle=60,
  seed=1,
  threshold=0.3
}
valDataInfo = mnist_cluttered.createData(valDataConfig)

local nVal = 1000
local valData = torch.FloatTensor(nVal, 1, valDataConfig.megapatch_w, valDataConfig.megapatch_w)
local valLabel = torch.FloatTensor(nVal, 10, valDataConfig.megapatch_w, valDataConfig.megapatch_w)
for i = 1,nVal do
  local observation, target = unpack(valDataInfo.nextExample())
  valData[i] = observation
  valLabel[i] = target
end

local testDataConfig = {
  datasetPath = 'mnist/test.t7',
  megapatch_w=64,
  nDigits=5,
  border=0,
  scale={0.5,1.25},
  angle=60,
  seed=1,
  threshold=0.3
}
testDataInfo = mnist_cluttered.createData(testDataConfig)

local nTest = 1000
local testData = torch.FloatTensor(nTest, 1, testDataConfig.megapatch_w, testDataConfig.megapatch_w)
local testLabel = torch.FloatTensor(nTest, 10, testDataConfig.megapatch_w, testDataConfig.megapatch_w)
for i = 1,nTest do
  local observation, target = unpack(testDataInfo.nextExample())
  testData[i] = observation
  testLabel[i] = target
end

local results = {}
results.train = {data=trainData, labels=trainLabel}
results.test = {data=testData, labels=testLabel}
results.val = {data=valData, labels=valLabel}
torch.save('mnist_segmentation.t7', results)
