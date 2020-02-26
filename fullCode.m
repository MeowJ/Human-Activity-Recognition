%% M L N C 18-19 coursework
%  CID: 1555404
close all
clear all
clc
load data; %numData = 24000, data_length = 65
[total SizeInput] = size(data);
meanv = mean(data(:,2:SizeInput));
stdv = std(data(:,2:SizeInput));
for i=2:SizeInput
    data(:,i) = (data(:,i)- meanv(i-1))./stdv(i-1);
end
% Another option
%[Xi - min(X)]/[max(X) - min(X)]
% [total SizeInput] = size(data);
% maxv = max(data);
% minv = min(data);
% for i=2:SizeInput
%     data(:,i) = (data(:,i)- minv(i))./(maxv(i)-min(i));
% end
%% shuffle data
shuffled = data(randperm(size(data,1)),:);
labels = shuffled(:,1);
inputs = shuffled(:,2:end);

all_test_acc = 0; % Array of test accuracy 
%Hyper_range = 30:5:120; % Value of Hyper parameter
epochMax = 25;
Tacclist = zeros(1,epochMax);

%% This for loop is used to tune the hyperparameters
%for hyperparameter=Hyper_range
SizeHidden1 = 30;
SizeHidden2 = 30;
SizeOutput = 5;
LearningRate = 0.1; % Learning rate
decay_rate = -0.005;

e = 0.1;
WeightHidden1 = rand(SizeInput, SizeHidden1) *e*2 - e;
WeightHidden2 = rand(SizeHidden1 + 1, SizeHidden2 + 1) *e*2 - e;
WeightOutput = rand(SizeHidden2 + 1, SizeOutput) *e*2 - e;
DeltaWH1 = zeros(SizeInput, SizeHidden1);
DeltaWH2 = zeros(SizeHidden1 + 1, SizeHidden2 + 1);
DeltaWO = zeros(SizeHidden2 + 1, SizeOutput);
Input = zeros(1, SizeInput);
Hidden1 = zeros(1, SizeHidden1 + 1);
Hidden2 = zeros(1, SizeHidden2 + 1);
Output = zeros(1, SizeOutput);

sigmoid = @(v) 1.0 ./ (1.0 + exp(-v));
get_output = @(i,wh1,wh2,wo) sigmoid(sigmoid([sigmoid([[inputs(i,:), -1] * wh1, -1]) * wh2]) * wo);

training_tag = 1:total*0.8;
generalization_tag = total*0.8:total*0.9;
validation_tag = total*0.9:total;

TtrainaccAll = 0;
ValidaccAll = 0;
tic;
for epoch  = 1:epochMax    
    fprintf('\nEpoch: %d Learning rate: %f\n', epoch, LearningRate);
	TrainMSE = 0;
	Trtainacc = 0;
	ValidMSE = 0;
	Validacc = 0;
	TestMSE = 0;
	Testacc = 0;
	v = 0;
	r = 0;
    tmpTacc = 0;
 	for i = training_tag
		% Feed forward
		Input = [inputs(i,:), -1];
		Hidden1 = sigmoid([Input * WeightHidden1, -1]);
        Hidden2 = sigmoid([Hidden1 * WeightHidden2]);
		Output = sigmoid(Hidden2 * WeightOutput);

		v = Output(labels(i));
		TrainMSE = TrainMSE + (1-v)^2;
		[v,r] = max(Output);
		if labels(i) == r
			Trtainacc = Trtainacc+1;
		end

		% Back-propagation and weight updates
		Desired = zeros(1,SizeOutput);
		Desired(labels(i)) = 1;
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
	TrainMSE = TrainMSE / size(training_tag,2);
	Trtainacc = Trtainacc / size(training_tag,2) * 100;

	for i = generalization_tag
		Output = get_output(i,WeightHidden1,WeightHidden2,WeightOutput);
		v = Output(labels(i));
		ValidMSE = ValidMSE + (1-v)^2;
		[v,r] = max(Output);
		if labels(i) == r
			Validacc = Validacc+1;
		end
    end
    LearningRate = LearningRate * exp(decay_rate*epoch);
	ValidMSE = ValidMSE / size(generalization_tag,2);
	Validacc = Validacc / size(generalization_tag,2) * 100;
    
    TtrainaccAll = [TtrainaccAll Trtainacc];
    ValidaccAll = [ValidaccAll Validacc];
    fprintf('TrainError: %f, TrainAcc: %f, ValidationError: %f, ValidationAcc: %f\n',TrainMSE,Trtainacc,ValidMSE,Validacc);
        
 end
toc

for i = validation_tag
	Output = get_output(i,WeightHidden1,WeightHidden2,WeightOutput);
	v = Output(labels(i));
	TestMSE = TestMSE + (1-v)^2;
	[v,r] = max(Output);
	if labels(i) == r
		Testacc = Testacc+1;
    end
end
TestMSE = TestMSE / size(validation_tag,2);
Testacc = Testacc / size(validation_tag,2) * 100;

fprintf('TestError: %f, TestAcc: %f\n',TestMSE,Testacc);
all_test_acc = [all_test_acc Testacc];
TtrainaccAll(1)=[];
Tacclist = [Tacclist; TtrainaccAll];
%end

%% plots

%plot Accuracy VS Learning rate
% figure
% all_test_acc(1)=[];
% plot(Hyper_range,all_test_acc)
% title('Accuracy VS Learning rate')
% xlabel('Learning rate')
% ylabel('Accuracy')
% grid on

% %plot Accuracy VS Number of epoches when learning rate is different
% figure
% ValidaccAll(1) = [];
% plot(1:epochMax,TtrainaccAll)
% hold on
% plot(1:epochMax,ValidaccAll)
% title('Accuracy VS Number of epoches')
% xlabel(' Number of epoches')
% ylabel('Accuracy')
% grid on
% ylim([90 100])

% %plot Accuracy VS Number of epoches to find the epoches number
% figure
% Tacclist(1,:) = [];
% plot(1:epochMax,Tacclist)
% title('Accuracy VS Number of epoches')
% xlabel(' Number of epoches')
% ylabel('Accuracy')
% grid on
% ylim([90 100])

%Plot confusion matrix
%figure
g1_train = labels(generalization_tag);
g2_train = 0;
for kk = generalization_tag
    ans = get_output(kk,WeightHidden1,WeightHidden2,WeightOutput);
    [v,r] = max(ans);
    g2_train = [g2_train r];
end
g2_train(1)=[];
g2_train = g2_train';
Confusion_Matrix = confusionmat(g1_train',g2_train)


