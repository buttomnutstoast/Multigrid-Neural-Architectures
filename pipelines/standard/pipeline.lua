EPOCH = OPT.epochNumber
for i=OPT.epochNumber, OPT.nEpochs do
    if OPT.train then train() end
    if OPT.eval and (i == OPT.nEpochs or i % OPT.nEpochsEval == 0) then eval() end
    if OPT.test and (i == OPT.nEpochs or i % OPT.nEpochsTest == 0) then test() end
    if (OPT.train and (i == OPT.nEpochs or i % OPT.nEpochsSave == 0)) then
        saveDataParallel(paths.concat(OPT.save, 'model_' .. EPOCH .. '.t7'), MODEL)
    end
    EPOCH = EPOCH + 1
end
