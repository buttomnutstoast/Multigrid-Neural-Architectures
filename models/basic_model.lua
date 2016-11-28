--[[
    Model Prototype includes functions:
    1. BASICNET.packages: required torch/custom packages
    2. BASICNET.createModel: definition of neural network model
    3. BASICNET.createCriterion: definition of loss criterion
    4. BASICNET.trainOutputInit, BASICNET.testOutputInit, BASICNET.evalOutputInit:
        initiate log table (the input argument-info) which is helpful to carry
        information between different iterations/epoches
    5. BASICNET.trainOutput, BASICNET.testOutput, BASICNET.evalOutput:
        perform computation of carried data in info table
    6. BASICNET.ftrain, BASICNET.ftest, BASICNET.feval:
        feed forward trianing/testing/evaluation
    7. BASICNET.btrain: backward propogation
    8. BASICNET.gradProcessing: post processing on gradients
    9. BASICNET.arguments: customized opts of the model
    10. BASICNET.trainRule: customized learning rate and weight decay
--]]

local BASICNET = {}

function BASICNET.packages()
end

function BASICNET.createModel(opt)
    error('BASICNET.createModel is called; you should implement your model')
end

function BASICNET.createCriterion()
    error('BASICNET.createCriterion is called; you should implement your criterion')
end

function BASICNET.trainOutputInit()
    error('BASICNET.trainOutputInit is called; you should implement your trainOutputInit')
end

function BASICNET.trainOutput(info, outputs, labelsCPU, err, iterSize)
    error('BASICNET.trainOutput is called; you should implement your trainOutput')
end

function BASICNET.testOutputInit()
    error('BASICNET.testOutputInit is called; you should implement your testOutputInit')
end

function BASICNET.testOutput(info, outputs, labelsCPU, err)
    error('BASICNET.testOutput is called; you should implement your testOutput')
end

function BASICNET.evalOutputInit()
    error('BASICNET.evalOutputInit is called; you should implement your evalOutputInit')
end

function BASICNET.evalOutput(info, outputs, labelsCPU, err)
    error('BASICNET.evalOutput is called; you should implement your evalOutput')
end

function BASICNET.ftrain(inputs, labels, model, criterion)
    local outputs = model:forward(inputs)
    local err = criterion:forward(outputs, labels)
    local gradOutputs = criterion:backward(outputs, labels)
    model:backward(inputs, gradOutputs)
    return outputs, err
end

function BASICNET.btrain(parameters, funcErrGradPa, optim, optimState)
    optim.sgd(funcErrGradPa, parameters, optimState)
end

function BASICNET.ftest(inputs, labels, model, criterion)
    outputs = model:forward(inputs)
    err = criterion:forward(outputs, labels)
    return outputs, err
end

function BASICNET.feval(inputs, labels, model, criterion)
    outputs = model:forward(inputs)
    err = criterion:forward(outputs, labels)
    return outputs, err
end

function BASICNET.gradProcessing(model, modelPa, modelGradPa, currentEpoch)
end

function BASICNET.arguments(cmd)
end

function BASICNET.trainRule(currentEpoch, opt)
    error('BASICNET.trainRule is called; you should manually assign'
        .. ' Learning Rate & Weight Decay or implement your trainRule')
end

return BASICNET
