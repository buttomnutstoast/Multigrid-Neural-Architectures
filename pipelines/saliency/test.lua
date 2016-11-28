require 'optim'
require 'os'
require 'utils/utilfuncs'

local hdf5 = require 'hdf5'

local predH5DB
function test()
    collectgarbage()
    print('==> doing epoch on testing data:')
    print("==> online epoch # " .. EPOCH)

    local batchSize = 1

    cutorch.synchronize()

    -- set the dropouts to evaluate mode
    MODEL:evaluate()

    -- open hdf5 db
    local dbName = paths.concat(OPT.save, 'testOutput.h5')
    predH5DB = hdf5.open(dbName, 'w')

    for i=1,NTEST do
        local indexStart = i
        local indexEnd = i
        local currentEpoch = EPOCH
        DONKEYS:addjob(
            -- work to be done by donkey thread
            function()
                local inputs, labels = testLoader:getInputs(
                    indexStart,
                    indexEnd,
                    currentEpoch
                    )
                return inputs, labels, testLoader.mask, indexStart
            end,
            -- callback that is run in the main thread once the work is done
            testBatch
            )
    end

    DONKEYS:synchronize()
    cutorch.synchronize()

    predH5DB:close()
    print('\n')

end -- of test()
------------------------------------------------------------------------
local function interestPoint(input)
    -- select 7 intrested points, 3 in the left and 3 in the right
    -- 1 in the center
    assert(input:size(1) == 1 and input:dim() == 3)
    local th = 0.4
    local colMax = input:max(2):squeeze():ge(th)
    local rowMax = input:max(3):squeeze():ge(th)
    if colMax:sum() < 2 or rowMax:sum() < 2 then return nil end

    -- get a rough bounding box
    local colSt, colEd, rowSt, rowEd
    for i = 1,colMax:size(1) do
        if colMax[i] == 1 then
            colSt = i
            break
        end
    end

    for i = colMax:size(1),1,-1 do
        if colMax[i] == 1 then
            colEd = i
            break
        end
    end

    for i = 1,rowMax:size(1) do
        if rowMax[i] == 1 then
            rowSt = i
            break
        end
    end

    for i = rowMax:size(1),1,-1 do
        if rowMax[i] == 1 then
            rowEd = i
            break
        end
    end

    local function helper(p, size, dp)
        p = p - dp <= 0 and dp + 1 or p
        p = p + dp > size and size - dp or p
        return p
    end

    -- get 7 points (each point is a 2x2 region)
    local points = torch.FloatTensor(7,unpack(input:size():totable()))
    points:zero()

    local x, y
    local dp = 1
    -- center point
    x = helper(math.floor((colSt+colEd)/2), input:size(3), dp)
    y = helper(math.floor((rowSt+rowEd)/2), input:size(2), dp)
    points[{{1},{},{y-dp,y+dp},{x-dp,x+dp}}] = 1
    -- side points
    for ix = 1,2 do
        x = helper(colSt + (ix-1)*(colEd-colSt), input:size(3), dp)
        for iy = 1,3 do
            y = helper(rowSt + math.ceil((rowEd-rowSt)*(iy-1)/2),
                       input:size(2), dp)
            local _n = (ix-1)*3+iy+1
            points[{{_n},{},{y-dp,y+dp},{x-dp,x+dp}}] = 1
        end
    end
    return points
end

local inputsGPUTable = {}
local inputs = nil

function testBatch(inputsCPU, labelsCPU, maskCPU, ind)

    utilfuncs.put2GPU(inputsCPU, inputsGPUTable)

    inputs = inputsGPUTable[1]
    inputsCPU = inputsCPU[1]

    labelsCPU = labelsCPU[1]

    local outputs = {}
    for i = 1,8 do
        local _is = inputs:size(1)
        local _s = math.floor(_is * (i-1) / 8) + 1
        local _e = i == 8 and _is or math.floor(_is * i / 8)
        outputs[i] = MODEL:forward(inputs[{{_s,_e}}]):float()
    end
    cutorch.synchronize()
    local outputsCPU = torch.cat(outputs, 1)

    -- get interested points
    local points = interestPoint(outputsCPU[1])
    -- skip this iteration if no interested points found
    if not points then return end 

    -- compute saliency map
    local threshold = 0.1
    local saliency = points.new():resizeAs(points):zero()
    local ori = outputsCPU[1]
    for i = 1,maskCPU:size(1) do
        local masked = torch.csub(outputsCPU[i+1], ori):abs()
        for j = 1,points:size(1) do
            local impact = torch.cmul(masked, points[j]):max()
            local isSalient = impact >= threshold and 1 or 0
            saliency[j]:add(maskCPU[i]*impact*isSalient)
        end
    end

    inputs = nil
    collectgarbage()


    local id = string.format('id%05d', ind)
    print(id)
    predH5DB:write(id..'/points', points)
    predH5DB:write(id..'/saliency', saliency)
    predH5DB:write(id..'/input', inputsCPU)
    predH5DB:write(id..'/output', outputsCPU)
end
