function label = ClassifyX(input, parameters)
    WeightHidden1 = parameters.WeightHidden1;
    WeightHidden2 = parameters.WeightHidden2;
    WeightOutput = parameters.WeightOutput;
    sigmoid = @(u) 1.0 ./ (1.0 + exp(-u));
    get_output = @(wh1,wh2,wo) sigmoid(sigmoid([sigmoid([[input, -1] * wh1, -1]) * wh2]) * wo);
    Output = get_output(WeightHidden1,WeightHidden2,WeightOutput);
    [maxP, label] = max(Output);
end


