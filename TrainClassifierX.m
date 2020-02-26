function parameters = TrainClassifierX(input, label)
[total SizeInput] = size(input);
% Hyperparameters
epochMax = 25; %Number of epochs
SizeHidden1 = 30; %Number of nodes on hidden layers
SizeHidden2 = 30;
SizeOutput = 5; %number of activity classes
LearningRate = 0.1; 
decay_rate = -0.005; %decay rate of learning rate 

% Use some robust parameter if input data haven't be preprocessed.
if max(mean(input))>10
    LearningRate = 0.0007;
    SizeHidden1 = 90;
    epochMax = 45;
end

% Weight between each two layers
% NOTE: the bias term is added into the last column of the weight matrixes
WeightHidden1 = rand(SizeInput + 1, SizeHidden1)*0.1*2 - 0.1;
WeightHidden2 = rand(SizeHidden1 + 1, SizeHidden2 + 1)*0.1*2 - 0.1;
WeightOutput = rand(SizeHidden2 + 1, SizeOutput)*0.1*2 - 0.1;
DeltaWH1 = zeros(SizeInput + 1, SizeHidden1);
DeltaWH2 = zeros(SizeHidden1 + 1, SizeHidden2 + 1);
DeltaWO = zeros(SizeHidden2 + 1, SizeOutput);

% Train the weight vaule iteratively, observe the trainning accuracy 
tic; %Check the time cost
sigmoid = @(u) 1.0 ./ (1.0 + exp(-u));
for epoch  = 1:epochMax
    Tacc = 0; % Training accuracy
    for i = 1:total
		% Feed forward
		Input = [input(i,:), -1];
		Hidden1 = sigmoid([Input * WeightHidden1, -1]);
        Hidden2 = sigmoid([Hidden1 * WeightHidden2]);
		Output = sigmoid(Hidden2 * WeightOutput);
        % Test trainning accuracy 
        v = Output(label(i));
		[v,r] = max(Output);
		if label(i) == r
			Tacc = Tacc+1;
		end

		% Back-propagation and weight updates (using gradient descent)
		Desired = zeros(1,SizeOutput);
		Desired(label(i)) = 1;
		ErrorOutput = Output.*(1-Output).*(Desired-Output);
		ErrorHidden2 = Hidden2.*(1-Hidden2).*(sum((WeightOutput.*ErrorOutput)'));
        ErrorHidden1 = Hidden1.*(1-Hidden1).*(sum((WeightHidden2.*ErrorHidden2)'));
		DeltaWO = LearningRate*(Hidden2' * ErrorOutput) + 0.7*DeltaWO;
		WeightOutput = WeightOutput + DeltaWO;
		DeltaWH2 = LearningRate*(Hidden1' * ErrorHidden2) + 0.7*DeltaWH2;
		WeightHidden2 = WeightHidden2 + DeltaWH2;
        DeltaWH1 = LearningRate*(Input' * ErrorHidden1(1,1:SizeHidden1)) + 0.7*DeltaWH1;
		WeightHidden1 = WeightHidden1 + DeltaWH1;
    end
    % reduce the learning rate with the increase of epoch
    LearningRate = LearningRate * exp(decay_rate*epoch);
    % Calculate and print test accuracy 
    Tacc = Tacc / total * 100;
    fprintf('Epoch: %d\tTrain Accuracy: %f\n',epoch,Tacc);
end
toc
% Wrap the parameters
parameters.WeightHidden1 = WeightHidden1;
parameters.WeightHidden2 = WeightHidden2;
parameters.WeightOutput = WeightOutput;
end