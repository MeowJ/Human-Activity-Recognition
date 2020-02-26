%% M L N C 18-19 coursework 
%  CID: 1555404
%  A 2-hidden-layer MLP classifier 
close all
clear all
clc
%% Data preprocess
% let the mean of each feature close to 0, make the variance of each
% feature close to 1.
% THIS STEP IS NECESSARY FOR 2-HIDDEN-LAYER MLP!CANNOT BE REMOVED!
load data; %numData = 24000, data_length = 65
[total SizeInput] = size(data);
mean = mean(data);
std = std(data);
for i=2:SizeInput
    data(:,i) = (data(:,i)- mean(i))./std(i);
end

%% shuffle data and split it for cross validation
% Split data into trainning data(0.9) and test data(0.1)
shuffled = data(randperm(size(data,1)),:);
labels = shuffled(:,1);
inputs = shuffled(:,2:end);
train_tag = 1:total*0.9;
test_tag = total*0.9:total;
train_data = inputs(train_tag,:);
train_label = labels(train_tag);
test_data = inputs(test_tag,:);
test_label = labels(test_tag);

%% Generate and unpack parameters
parameters = TrainClassifierX(train_data, train_label);
WeightHidden1 = parameters.WeightHidden1; %unpack parameters
WeightHidden2 = parameters.WeightHidden2;
WeightOutput = parameters.WeightOutput;

%% Test the accuracy of the classifier
% Find Trainning result 
Train_predict = 0;
for n = 1:length(train_label)
    r = ClassifyX(train_data(n,:), parameters);
    Train_predict = [Train_predict r];
end

% Find Testing accuracy 
Test_predict = 0;
Test_acc = 0;
for n = 1:length(test_tag)
    r = ClassifyX(test_data(n,:), parameters);
    Test_predict = [Test_predict r];
    if test_label(n) == r
		Test_acc = Test_acc+1;
    end
end
Test_acc = Test_acc / size(test_tag,2) * 100;

% Print Accuracy and Confusion Matrix
Test_predict(1)=[];Train_predict(1)=[];
Train_Confusion_Matrix = confusionmat(train_label,Train_predict)
Test_Confusion_Matrix = confusionmat(test_label,Test_predict)
fprintf('Test accuracy: %f\n',Test_acc);
