-------------------------------
-- Different cluttering task --
-------------------------------
local M = {}

local function checkConfig(config)
  assert(config.datasetPath, 'illegal config')
  assert(config.megapatch_w, 'illegal config')
  assert(config.border, 'illegal config')
  assert(config.nDigits, 'illegal config')
  assert(config.nClasses, 'illegal config')
  assert(config.threshold, 'illegal config')
  assert(config.scale, 'illegal config')
  assert(config.angle, 'illegal config')
  assert(config.affine_angle, 'illegal config')
  assert(config.seed, 'illegal config')
  assert(config.task, 'illegal config')
end

function M.task(config, dataset)
    M._step = M._step or dataset.data:size(1)
    M._perm = M._perm or torch.Tensor()
    M._obs = M._obs or torch.Tensor()
    M._target = M._target or torch.Tensor()
    return M[config.task](config, dataset)
end


function M.segmentation(config, dataset)
  local tf = require 'utils/transform'
  local sp = require 'utils/position'

  -- Puts the sprite on a random position inside of the obs.
  -- The observation should have intensities in the [0, 1] range.
  local function placeSprite(obs, target, sprite, class, y, x)
    local spriteH = sprite:size(2)
    local spriteW = sprite:size(3)

    local subTensor = obs[{{}, {y, y + spriteH - 1}, {x, x + spriteW - 1}}]
    local subTensorTarget = target[{{class}, {y, y + spriteH - 1}, {x, x + spriteW - 1}}]
    subTensor:add(sprite)
    subTensorTarget:add(sprite)
    -- Keeping the values in the [0, 1] range.
    subTensor:apply(function(x)
      if x > 1 then
        return 1
      end
      if x < 0 then
        return 0
      end
      return x
    end)
  end


  local nExamples = dataset.data:size(1)
  M._obs:resize(dataset.data[1]:size(1), config.megapatch_w, config.megapatch_w)
  M._target:resize(config.nClasses, config.megapatch_w, config.megapatch_w)
  M._obs:zero()
  M._target:zero()

  local nDigits = torch.ceil(torch.normal(config.nDigits, 0.5))
  for i = 1, nDigits do
    M._step = M._step + 1
    if M._step > nExamples then
      torch.randperm(M._perm, nExamples)
      M._step = 1
    end

    local sprite = dataset.data[M._perm[M._step]]
    local selectedDigit = dataset.labels[M._perm[M._step]][1] + 1
    if config.angle ~= 0 then sprite = tf.rotate(sprite, config.angle) end
      
    local rescaleSprite
    local y, x
    repeat
      rescaleSprite = tf.rescale(sprite, config.scale)
      rescaleSprite = tf.normalize(rescaleSprite)
      y, x = sp.sample('overlap_constraint', M._obs, rescaleSprite, config.border)
    until y and x
    placeSprite(M._obs, M._target, rescaleSprite, selectedDigit, y, x)
  end
        
  M._target = torch.ge(M._target, config.threshold):typeAs(M._target)

  return {M._obs, M._target}
end

-- Puts the sprite on a random position inside of the obs.
-- The observation should have intensities in the [0, 1] range.
local function placeSprite(canvas, sprite, y, x)
  local spriteH = sprite:size(2)
  local spriteW = sprite:size(3)

  local subTensor = canvas[{{}, {y, y+spriteH-1}, {x, x+spriteW-1}}]
  subTensor:add(sprite)
  -- Keeping the values in the [0, 1] range.
  subTensor:apply(function(x)
    if x > 1 then
      return 1
    end
    if x < 0 then
      return 0
    end
    return x
  end)
end


function M.spatial_transform(config, dataset)
  local tf = require 'utils/transform'
  local sp = require 'utils/position'

  local nExamples = dataset.data:size(1)
  M._obs:resize(dataset.data[1]:size(1), config.megapatch_w, config.megapatch_w)

  local tgSize = config.megapatch_w
  --tgSize = tgSize >=  dataset.data[1]:size(2) and tgSize or dataset.data[1]:size(2)
  --tgSize = tgSize >=  dataset.data[1]:size(3) and tgSize or dataset.data[1]:size(3)
  M._target:resize(dataset.data[1]:size(1), tgSize, tgSize)

  M._obs:zero()
  M._target:zero()

  local nDigits = config.nDigits
  for i = 1, nDigits do
    M._step = M._step + 1
    if M._step > nExamples then
      torch.randperm(M._perm, nExamples)
      M._step = 1
    end

    local sprite = dataset.data[M._perm[M._step]]
    local ground_truth = sprite:clone()
    if config.angle ~= 0 then sprite = tf.rotate(sprite, config.angle) end
    if config.affine_angle ~= 0 then sprite = tf.affine(sprite, config.affine_angle) end
      
    local rescaleSprite
    local y, x
    repeat
      rescaleSprite = tf.rescale(sprite, config.scale)
      rescaleSprite = tf.normalize(rescaleSprite)
      y, x = sp.sample('uniform', M._obs, rescaleSprite, config.border)
    until y and x
    placeSprite(M._obs, rescaleSprite, y, x)


    local gty, gtx = sp.sample('center', M._target, ground_truth, config.border)
    placeSprite(M._target, ground_truth, gty, gtx)
  end
        
  M._target = torch.ge(M._target, config.threshold):typeAs(M._target)

  return {M._obs, M._target}
    
end

function M.centered_transform(config, dataset)
  local tf = require 'utils/transform'
  local sp = require 'utils/position'

  local nExamples = dataset.data:size(1)
  M._obs:resize(dataset.data[1]:size(1), config.megapatch_w, config.megapatch_w)

  local tgSize = config.megapatch_w
  M._target:resize(dataset.data[1]:size(1), tgSize, tgSize)

  M._obs:zero()
  M._target:zero()

  local nDigits = config.nDigits
  for i = 1, nDigits do
    M._step = M._step + 1
    if M._step > nExamples then
      torch.randperm(M._perm, nExamples)
      M._step = 1
    end

    local sprite = dataset.data[M._perm[M._step]]
    local ground_truth = sprite:clone()
    if config.angle ~= 0 then sprite = tf.rotate(sprite, config.angle) end
    if config.affine_angle ~= 0 then sprite = tf.affine(sprite, config.affine_angle) end
      
    local rescaleSprite
    local y, x
    repeat
      rescaleSprite = tf.rescale(sprite, config.scale)
      rescaleSprite = tf.normalize(rescaleSprite)
      y, x = sp.sample('center', M._obs, rescaleSprite, config.border)
    until y and x
    placeSprite(M._obs, rescaleSprite, y, x)


    local gty, gtx = sp.sample('center', M._target, ground_truth, config.border)
    placeSprite(M._target, ground_truth, gty, gtx)
  end
        
  M._target = torch.ge(M._target, config.threshold):typeAs(M._target)

  return {M._obs, M._target}

end

return M.task
