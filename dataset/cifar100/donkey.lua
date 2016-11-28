require 'image'
local tf = require 'utils/transforms'

paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(OPT.cache, 'trainCache.t7')
local testCache = paths.concat(OPT.cache, 'testCache.t7')

-- Check for existence of OPT.data
if not os.execute('cd ' .. OPT.data) then
    error(("could not chdir to '%s'"):format(OPT.data))
end

-- channel-wise mean and std
local meanstd = {
    mean = {129.3, 124.1, 112.4},
    std  = {68.2,  65.4,  70.4},
}
--------------------------------------------------------------------------------
--[[
   Section 0: Create image loader functions each for training ane testing,
   the function in training upscale the shorter side to loadSize, however,
   the one in testing upscale the longer side.
--]]

local function loadImage(img)
    return img:view(3,32,32)
end


-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean, std
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, img)
    local input = loadImage(img)
    local out = tf.ColorNormalize(meanstd)(input)

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
    local out = tf.ColorNormalize(meanstd)(input)

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
