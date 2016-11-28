local mnist_cluttered = require 'mnist_cluttered'

local trainDataConfig = {
  datasetPath = 'mnist/train.t7',
  task = 'spatial_transform',
  megapatch_w = 64,
  nDigits = 1,
  border = 0,
  scale = {0.5,1.5},
  angle = 60,
  affine_angle = 60,
  seed = 1,
  threshold = 0.3
}
trainDataInfo = mnist_cluttered.createData(trainDataConfig)

local nTrain = 50000
local trainData = torch.FloatTensor(nTrain, 1, trainDataConfig.megapatch_w, trainDataConfig.megapatch_w)
local trainLabel = torch.FloatTensor(nTrain, 1, trainDataConfig.megapatch_w, trainDataConfig.megapatch_w)
for i = 1,nTrain do
  local observation, target = unpack(trainDataInfo.nextExample())
  trainData[i] = observation
  trainLabel[i] = target
end

local valDataConfig = {
  datasetPath = 'mnist/valid.t7',
  task = 'spatial_transform',
  megapatch_w = 64,
  nDigits = 1,
  border = 0,
  scale = {0.5,1.5},
  angle = 60,
  affine_angle = 60,
  seed = 1,
  threshold = 0.3
}
valDataInfo = mnist_cluttered.createData(valDataConfig)

local nVal = 10000
local valData = torch.FloatTensor(nVal, 1, valDataConfig.megapatch_w, valDataConfig.megapatch_w)
local valLabel = torch.FloatTensor(nVal, 1, valDataConfig.megapatch_w, valDataConfig.megapatch_w)
for i = 1,nVal do
  local observation, target = unpack(valDataInfo.nextExample())
  valData[i] = observation
  valLabel[i] = target
end

local testDataConfig = {
  datasetPath = 'mnist/test.t7',
  task = 'spatial_transform',
  megapatch_w = 64,
  nDigits = 1,
  border = 0,
  scale = {0.5,1.5},
  angle = 60,
  affine_angle = 60,
  seed = 1,
  threshold = 0.3
}
testDataInfo = mnist_cluttered.createData(testDataConfig)

local nTest = 10000
local testData = torch.FloatTensor(nTest, 1, testDataConfig.megapatch_w, testDataConfig.megapatch_w)
local testLabel = torch.FloatTensor(nTest, 1, testDataConfig.megapatch_w, testDataConfig.megapatch_w)
for i = 1,nTest do
  local observation, target = unpack(testDataInfo.nextExample())
  testData[i] = observation
  testLabel[i] = target
end

local results = {}
results.train = {data=torch.cat({trainData,valData},1), labels=torch.cat({trainLabel,valLabel},1)}
results.test = {data=testData, labels=testLabel}
torch.save('mnist_spatial_transform.t7', results)
