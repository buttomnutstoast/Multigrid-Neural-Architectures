--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

require 'torch'
require 'image'

local M = {}

-- Copies values from src to dst.
local function update(dst, src)
  for k, v in pairs(src) do
    dst[k] = v
  end
end

-- Copies the config. An error is raised on unknown params.
local function updateDefaults(dst, src)
  for k, v in pairs(src) do
    if dst[k] == nil then
      error("unsupported param: " .. k)
    end
  end
  update(dst, src)
end

local function loadDataset(path)
  local dataset = torch.load(path)
  dataset.data = dataset.data:type(torch.Tensor():type())
  collectgarbage()
  dataset.data:mul(1/dataset.data:max())

  if dataset.data[1]:dim() ~= 3 then
    local sideLen = math.sqrt(dataset.data[1]:nElement())
    dataset.data = dataset.data:view(dataset.data:size(1), 1, sideLen, sideLen)
  end

  assert(dataset.labels:min() == 0, "expecting zero-based labels")
  return dataset
end

-- Return a list with pointers to selected examples.
local function selectSamples(examples, nSamples)
  local nExamples = examples:size(1)
  local samples = {}
  for i = 1, nSamples do
    samples[i] = examples[torch.random(1, nExamples)]
  end
  return samples
end

-- Returns a map from {smallerDigit, biggerOrEqualDigit}
-- to an input in the softmax output.
local function createIndexMap(n, k)
  assert(k == 2, "expecting k=2")
  local indexMap = torch.Tensor(n, n):fill(0/0)
  local nextIndex = 1
  for i = 1, n do
    for j = i, n do
      indexMap[i][j] = nextIndex
      nextIndex = nextIndex + 1
    end
  end
  assert(k == 2 and nextIndex - 1 == (n * (n + 1))/2, "wrong count for k=2")
  return indexMap
end

-- The task is a classification of MNIST digits.
-- Each training example has a MNIST digit placed on a bigger black background.
function M.createData(extraConfig)
  local config = {
    datasetPath = 'mnist/train.t7',
    -- The size of the background.
    megapatch_w = 28,
    -- The width of a black border.
    border = 0,
    -- The number of digits in on image.
    nDigits = 1,
    -- The number of digit classes.
    nClasses = 10,
    -- The threshold for digit classes.
    threshold = 0.1,
    -- The range for rescaling digit.
    scale = {0.9,1.1},
    -- The angle for rotation.
    angle = 0,
    -- The angle for affine transform.
    affine_angle = 0,
    -- The random seed.
    seed = 100,
    -- The task.
    task = 'segmentation',
  }
  updateDefaults(config, extraConfig)
  torch.manualSeed(config.seed)

  local dataset = loadDataset(config.datasetPath)
  assert(dataset.labels:max() < config.nClasses, "expecting labels from {0, .., nClasses - 1}")

  local task = require 'utils/task'
  local function nextExample()
    local results = task(config, dataset)

    return results
  end

  return {
    nextExample = nextExample,
  }
end

return M
