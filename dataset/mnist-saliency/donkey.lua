require 'image'
local tf = require 'utils/transforms'

paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local testCache = paths.concat(OPT.cache, 'testCache.t7')

-- Check for existence of OPT.data
if not os.execute('cd ' .. OPT.data) then
    error(("could not chdir to '%s'"):format(OPT.data))
end

-- channel-wise mean and std
local meanstd = {
    mean = {0.1141},
    std  = {0.2746},
}

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]
function ColorNormalize(meanstd)
    return function(img)
        img = img:clone()
        img:add(-meanstd.mean[1])
        img:div(meanstd.std[1])
        return img
    end
end


local testHook = function(self, img)
    return tf.Compose{
        tf.Occlusion(self.mask),
        ColorNormalize(meanstd)
    }(img)
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
