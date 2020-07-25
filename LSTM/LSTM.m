clearvars
close all
clc
rng('shuffle'); %shuffling the random number generator
%LSTM for turbofan RUL prediction
%
%By: Abubakar Bala
%Topic: LSTM for RUL prediction usign matlab tool box
%Started on: 24/07/2020 :DD/MM/YYYY





for run = 1:4
    
    switch(run)
        
        case 1
            veryRawData = load('FD001_edited.txt');
            trainLen = 14000;
            testLen = 6000;
            
        case 2
            
            veryRawData = load('FD002_edited.txt');
            trainLen = 35000;
            testLen = 15000;
            
        case 3
            
            veryRawData = load('FD003_edited.txt');
            trainLen = 14000;
            testLen = 6000;
            
        case 4
            
            veryRawData = load('FD004_edited.txt');
            trainLen = 35000;
            testLen = 15000;
            
    end
    
    
    
    
    
    %Selecting the best signals
    maxSig = 14;
    rawData = veryRawData(:,[7,8,9,12,13,14,16,17,18,19,20,22,25,26]);
    
    %normalizing the targets
    targets = veryRawData(:,2);
   
    
    
    normData = zeros(size(rawData,1),maxSig);
    for sig = 1: maxSig
        column = rawData(:,sig);
        %newColumn = (2*(column - min(column))/(max(column) - min(column))) -1;
        normData(:,sig) = rescale(column,-1,1);
        %mu = mean(column);
        %sigma = std(column);
        %normData(:,sig) =  (column - mu) ./ sigma;
    end
    
    
    
    %smoothing and removing noise from data
    
    normData = smoothdata(normData,'gaussian');
    
    normData = [veryRawData(:,1), veryRawData(:,2), normData];
    
    newTrainLen = round(0.7 * trainLen);
    validLen = round(0.3*trainLen);
    
    [XTrain,YTrain] = trainPrepare(normData,newTrainLen);
    [XValid, YValid] = validPrepare(normData,validLen,newTrainLen);
    [XTest,YTest] = testPrepare(normData,testLen,trainLen);
    
    maxTrial = 10;
    mseTestTrial = zeros(maxTrial,1);
    timing = zeros(maxTrial,1);
    
    for trial = 1: maxTrial
        tic;
        
        
        numResponses = 1;
        numHiddenUnits = 256;
        numFeatures = 14;
        
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits,'OutputMode','sequence')
            fullyConnectedLayer(50)
            dropoutLayer(0.5)
            fullyConnectedLayer(numResponses)
            regressionLayer];
        
        
        options = trainingOptions('adam', ...
            'MaxEpochs',250, ...
            'MiniBatchSize',20, ...
            'ValidationData' , {XValid ,YValid},...
           'ValidationFrequency',50,...
            'ValidationPatience',5,...
            'ValidationFrequency',50,...
            'ExecutionEnvironment','cpu',...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.01, ...
            'Shuffle','never',...
            'Plots','training-progress',...
            'Verbose',0);
        
        
        net = trainNetwork(XTrain,YTrain,layers,options);
        
        
        
        
        YPred = predict(net,XTest,'MiniBatchSize',1)';
        
        %unrolling the prediction cell
        numCell = length(YPred);
        k = 1;
        YPredMat = [];
        for i = 1: numCell
            present =  YPred{i};
            presLen = length(present);
            if k == 1
                YPredMat(k:presLen) = present;
            else
                YPredMat(k:presLen + k-1) = present;
            end
            k = k + presLen;
        end
        
        %finding the test MSE
        yReal = targets(trainLen+1:trainLen + testLen);
        squareErrors = (YPredMat'-yReal).^2;
        mseTestTrial(trial) = sum(squareErrors)/length(yReal);
        timing(trial)  = toc;        
        
       
        
    end
    
    
    
    
    switch(run)
        
        case 1
            
            save('lstm_FD001','timing','mseTestTrial');          
            
            
        case 2
            
            
            save('lstm_FD002','timing','mseTestTrial');       
        case 3
            
            
            save('lstm_FD003','timing','mseTestTrial');       
            
        case 4
            
            save('lstm_FD004','timing','mseTestTrial');             
            
    end
    
    
end
