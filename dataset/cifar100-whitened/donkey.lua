require 'image'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(OPT.cache, 'trainCache.t7')
local testCache = paths.concat(OPT.cache, 'testCache.t7')
local meanstdCache = paths.concat(OPT.cache, 'meanstdCache.t7')

-- Check for existence of OPT.data
if not os.execute('cd ' .. OPT.data) then
    error(("could not chdir to '%s'"):format(OPT.data))
end

local loadSize   = {3, OPT.imageSize, OPT.imageSize}
local sampleSize = {3, OPT.imageCrop, OPT.imageCrop}

--------------------------------------------------------------------------------
--[[
   Section 0: Create image loader functions each for training ane testing,
   the function in training upscale the shorter side to loadSize, however,
   the one in testing upscale the longer side.
--]]
local function affineImg(img)
    local theta = torch.uniform(-math.pi/6, math.pi/6)
    local tan = torch.tan(theta)
    local diag = {tan, 0}
    local order = torch.randperm(2)
    local affineMat = torch.FloatTensor({{1, diag[order[1]]}, {diag[order[2]], 1}})
    img = image.affinetransform(img:double(), affineMat)
    return img
end

local function rotateImg(img)
    local theta = torch.uniform(-math.pi/6, math.pi/6)
    local sin = torch.sin(theta)
    local cos = torch.cos(theta)
    local affineMat = torch.FloatTensor({{cos, sin}, {-sin, cos}})
    -- better than using image.rotate (we don't want any area filled with 0)
    img = image.affinetransform(img:double(), affineMat)
    return img
end

local function resizeimg(img)
    -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
    local sideLen = math.min(img:size(3), img:size(2))
    assert(sideLen > 0, 'side length of input image should be larger than 0')

    local ratio = {-1, img:size(2)/sideLen, img:size(3)/sideLen}
    img = image.scale(img, loadSize[3]*ratio[3], loadSize[2]*ratio[2])
    return img
end

local function randCropImg(img)
    local iW = img:size(3)
    local iH = img:size(2)

    -- do random crop
    local oW = sampleSize[3]
    local oH = sampleSize[2]
    local h1 = math.floor(torch.uniform(1, iH-oH))
    local w1 = math.floor(torch.uniform(1, iW-oW))

    local img = image.crop(img, w1, h1, w1 + oW, h1 + oH)
    assert(img:size(3) == oW)
    assert(img:size(2) == oH)
    return img
end

local function centerCropImg(img)
    local oH = sampleSize[2]
    local oW = sampleSize[3]
    local iW = img:size(3)
    local iH = img:size(2)
    local w1 = math.ceil((iW-oW)/2)
    local h1 = math.ceil((iH-oH)/2)
    local img = image.crop(img, w1, h1, w1+oW, h1+oH) -- center patch
    return img
end

local function dataAug(img)
    -- img: RGB image
    if OPT.dataAug == 'affine' then
        img = affineImg(img)
    elseif OPT.dataAug == 'rotate' then
        img = rotateImg(img)
    end
    img = resizeimg(img)
    return img
end


local function colorspace(img)
    if OPT.colorspace == 'rgb' and img:dim() == 3 then
        return img
    elseif OPT.colorspace == 'bgr' and img:dim() == 3 then
        local bgr = img.new():resizeAs(img):zero()
        for ch = 1,3 do bgr[ch]:copy(img[4-ch]) end
        return bgr
    elseif OPT.colorspace == 'gray'then
        if img:dim() == 2 then
            img:resize(1, img:size(1), img:size(2))
        elseif img:dim() == 3 then
            img = image.rgb2y(img)
        else
            error('illegal number of image channel: ' .. img:dim())
        end
        return img
    else
        error('not implemented colorspace processing!!')
    end
end

local function loadImage(img)
    local img = img:view(3,32,32)
    img = colorspace(img)
    return dataAug(img)
end


--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, img)
    collectgarbage()
    local input = loadImage(img)
    local out = randCropImg(input)
    -- do hflip with probability 0.5
    if torch.uniform() > 0.5 then out = image.hflip(out) end

    return out
end

if paths.filep(trainCache) then
    print('Loading train metadata from cache')
    trainLoader = torch.load(trainCache)
    assert(trainLoader.path == paths.concat(OPT.data),
        'cached files dont have the same path as OPT.data.'
        .. 'Remove your cached files at: '
        .. trainCache .. ' and rerun the program')
else
    print('Creating train metadata')
    trainLoader = dataLoader{
        path = paths.concat(OPT.data),
        protocol = 'train',
    }
    torch.save(trainCache, trainLoader)
end
trainLoader.sampleHookTrain = trainHook
collectgarbage()

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
local testHook = function(self, img)
    collectgarbage()
    local input = loadImage(img)
    local crop_input = centerCropImg(input)

    -- padding
    local out = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3]):fill(0)
    out[{{}, {1, crop_input:size(2)}, {1, crop_input:size(3)}}] = crop_input

    return out
end

if paths.filep(testCache) then
    print('Loading test metadata from cache')
    testLoader = torch.load(testCache)
    assert(testLoader.path == paths.concat(OPT.data),
        'cached files dont have the same path as OPT.data.'
        .. 'Remove your cached files at: '
        .. testCache .. ' and rerun the program')
else
    print('Creating test metadata')
    testLoader = dataLoader{
        path = paths.concat(OPT.data),
        protocol = 'test',
    }
    torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
collectgarbage()
-- End of test loader section
