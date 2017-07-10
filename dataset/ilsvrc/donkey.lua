------------------------------------------
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
-- borrow some codes from https://github.com/facebook/fb.resnet.torch
------------------------------------------
require 'image'
local tf = require 'utils/transforms'

paths.dofile('dataset.lua')

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(OPT.cache, 'trainCache.t7')
local testCache = paths.concat(OPT.cache, 'testCache.t7')

-- Check for existence of OPT.data
if not os.execute('cd ' .. OPT.data) then
    error(("could not chdir to '%s'"):format(OPT.data))
end

-- Computed from random subset of ImageNet training images
local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}

local pca = {
    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
    eigvec = torch.Tensor{
        { -0.5675,  0.7192,  0.4009 },
        { -0.5808, -0.0045, -0.8140 },
        { -0.5836, -0.6948,  0.4203 },
    },
}
-------------------------------------------
-- load image from different data format --
-------------------------------------------
local function loadImage(path)
    local ok, input = pcall(function()
    return image.load(path, 3, 'float')
    end)

    -- Sometimes image.load fails because the file extension does not
    -- match the image format. In that case, use image.decompress on
    -- a ByteTensor.
    if not ok then
        local f = io.open(path, 'r')
        assert(f, 'Error reading: ' .. tostring(path))
        local data = f:read('*a')
        f:close()

        local b = torch.ByteTensor(string.len(data))
        ffi.copy(b:data(), data, b:size(1))

        input = image.decompress(b, 3, 'float')
    end
    return input
end


--------------------------------
-- Create a train data loader --
--------------------------------
local trainHook = function(self, path)
    local input = loadImage(path)
    local out = tf.Compose{
        tf.RandomSizedCrop(224),
        tf.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
        }),
        tf.Lighting(0.1, pca.eigval, pca.eigvec),
        tf.ColorNormalize(meanstd),
        tf.HorizontalFlip(0.5),
    }(input)

    return out
end

if paths.filep(trainCache) then
    print('Loading train metadata from cache')
    trainLoader = torch.load(trainCache)
    assert(trainLoader.paths[1] == paths.concat(OPT.data, 'train'),
        'cached files dont have the same path as OPT.data.'
        .. 'Remove your cached files at: '
        .. trainCache .. ' and rerun the program')
else
    print('Creating train metadata')
    trainLoader = dataLoader{
        paths = {paths.concat(OPT.data, 'train')},
        split = 100,
        verbose = true,
    }
    torch.save(trainCache, trainLoader)
end
trainLoader.sampleHookTrain = trainHook
collectgarbage()

-------------------------------
-- Create a test data loader --
-------------------------------
local testHook = function(self, path)
    local input = loadImage(path)
    local Crop = OPT.tenCrop and tf.TenCrop or tf.CenterCrop
    local out = tf.Compose{
        tf.Scale(256),
        tf.ColorNormalize(meanstd),
        Crop(224),
    }(input)

    return out
end

if paths.filep(testCache) then
    print('Loading test metadata from cache')
    testLoader = torch.load(testCache)
    assert(testLoader.paths[1] == paths.concat(OPT.data, 'val'),
        'cached files dont have the same path as OPT.data.'
        .. 'Remove your cached files at: '
        .. testCache .. ' and rerun the program')
else
    print('Creating test metadata')
    testLoader = dataLoader{
        paths = {paths.concat(OPT.data, 'val')},
        split = 0,
        verbose = true,
        forceClasses = trainLoader.classes
    }
    torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
collectgarbage()
