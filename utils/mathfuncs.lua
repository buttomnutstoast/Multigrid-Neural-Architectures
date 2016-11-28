mathfuncs = {}

function mathfuncs.topK(prediction, target, K)
    local acc = 0
    local _,prediction_sorted = prediction:sort(2, true) -- descending
    local batch_size = prediction:size(1)
    for i=1,batch_size do
        for j=1,K do
            if prediction_sorted[i][j] == target[i] then
                acc = acc + 1
                break
            end
        end
    end
    return acc / batch_size
end

local function iu(data, label)
    -- true positive
    local tp = data:eq(1):cmul(label:eq(1)):float():sum()
    -- false positive and false negative
    local fp_fn = data:ne(label):float():sum()
    -- iu = tp / (tp + fp + fn)
    local iu = tp / (tp + fp_fn)
    return iu
end

function mathfuncs.mnistIU(prediction, target, threshold)
    local prediction_ = prediction:ge(threshold):float()
    --local acc = iu(prediction_, target)
    local acc = 0

    for i = 1,target:size(1) do
        local nClass = 0
        for j = 1,target:size(2) do
            if target[{{i},{j},{},{}}]:eq(1):any() then
                nClass = nClass + 1
            end
        end
        acc = acc + iu(prediction_[i], target[i]) / nClass
    end
    acc = acc / target:size(1)
    return acc
end

return mathfuncs
