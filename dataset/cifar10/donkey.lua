require 'image'
transform = require 'utils/transforms'
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

local function resizeImage(img)
    return img:view(3,32,32)
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local meanstd = {
    mean = {125.3, 123.0, 113.9},
    std  = {63.0,  62.1,  66.7},
}

--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, img)
    local input = resizeImage(img)
    local out = transform.Compose{
        transform.ColorNormalize(meanstd),
        transform.HorizontalFlip(0.5),
        transform.RandomCrop(32,4),
    }

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

-- do some sanity checks on trainLoader
do
    local class = trainLoader.imageClass
    local nClasses = #trainLoader.classes
    assert(class:max() <= nClasses, "class logic has error")
    assert(class:min() >= 1, "class logic has error")

end

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
testHook = function(self, img)
    local input = resizeImage(img)
    local out = transform.ColorNormalize(meanstd)

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

