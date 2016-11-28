local ConcatUnet, Parent = torch.class('nn.ConcatUnet', 'nn.Module')

function ConcatUnet:__init()
    Parent.__init(self)
end

function ConcatUnet:updateOutput(input)
    -- input: {shortcut, sub_unet}
    -- shortcut = {t1,t2,...}
    -- subnet = {p1,p2,...}
    -- output: {{t1,p1},{t2,p2},...}

    self.output = {}
    assert(#input == 2, 'input should be {shortcut, sub_net}')
    assert(#input[1] >= #input[2], '#shortcut should >= #sub_net')

    local nEntry = #input[1]
    for i = 1,nEntry do
        self.output[i] = {}
        for j = 1,#input do
            self.output[i][j] = input[j][i]
        end
    end
    return self.output
end

function ConcatUnet:updateGradInput(input, gradOutput)
    self.gradInput = {}
    local nEntry = #input[1]
    for j = 1,#input do
        self.gradInput[j] = {}
        for i = 1,nEntry do
            self.gradInput[j][i] = gradOutput[i][j]
        end
    end
    return self.gradInput
end
