local pipDir = paths.concat('pipelines', OPT.pipeline) -- customized pipline
local stdPipDir = paths.concat('pipelines', 'standard') -- standard pipeline

-- load in functions for training, testing and evaluation
if OPT.train then
    if not paths.filep(paths.concat(pipDir, 'train.lua')) then
        print("it seems you don't have customized train.lua, import standard/train.lua")
        paths.dofile(paths.concat(stdPipDir, 'train.lua'))
    else
        paths.dofile(paths.concat(pipDir, 'train.lua'))
    end
end

if OPT.eval then
    if not paths.filep(paths.concat(pipDir, 'eval.lua')) then
        print("it seems you don't have customized eval.lua, import standard/eval.lua")
        paths.dofile(paths.concat(stdPipDir, 'eval.lua'))
    else
        paths.dofile(paths.concat(pipDir, 'eval.lua'))
    end
end

if OPT.test then
    if not paths.filep(paths.concat(pipDir, 'test.lua')) then
        print("it seems you don't have customized test.lua, import standard/test.lua")
        paths.dofile(paths.concat(stdPipDir, 'test.lua'))
    else
        paths.dofile(paths.concat(pipDir, 'test.lua'))
    end
end

-- run the pipeline
paths.dofile(paths.concat(pipDir, 'pipeline.lua'))