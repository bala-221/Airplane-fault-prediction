
clearvars
close all
clc
rng('shuffle'); %shuffling the random number generator

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
    end
    
    %smoothing and removing noise from data
    
    normData = smoothdata(normData,'gaussian');
    
 
    resSize = 1500;
    resConn = 0.5;
    inputSize = maxSig; %+ size(operatingConditions,2);
    outputSize = 1;
    leaky = 0.3;
    initLen = 300;
    
    
    
    maxTrial = 10;
    cost = zeros(maxTrial,1);
    mseTestTrial = zeros(maxTrial,1);
    cellWeights = cell(maxTrial,3);
    timing = zeros(maxTrial,1);
    spectralRadius = 1;
    
    
    
    
    for trial = 1: maxTrial
        
        tic;            
        
        inputWeights = rand(resSize,1+inputSize)*(2) -1;%selectign the range as [-1, +1]
        
        
        resWeights =  sprand(resSize,resSize,resConn);
        
        
        resWghtsMask = (resWeights~=0);
        %resWeights(resWghtsMask) = (resWeights(resWghtsMask)-0.5); %#ok<SPRIX>
        resWeights(resWghtsMask) = (resWeights(resWghtsMask)*2 -1); %#ok<SPRIX>
        
        
        %checking for ESP
        opt.disp = 0;
        rhoW = abs(eigs(resWeights ,1,'LM',opt));
        
        value = 25;
        
        while  isnan(rhoW)
            rhoW = abs(eigs(resWeights ,1,'LM','SubspaceDimension',value));
            value = value + 3;
        end
        
        resWeights  = resWeights .* (spectralRadius/rhoW);        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        reservAct = zeros(1+inputSize+resSize,trainLen-initLen);
        
        x = zeros(resSize,1); %initial instance of reservoir states
        
        for t = 1:trainLen
            u = normData(t,:)';
            x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x );
            if t > initLen
                reservAct(:,t-initLen) = [1;u;x];
            end
        end
        
        %Wout finding
        yTarget = targets(initLen+1:trainLen);
        
        outputWeights = yTarget'*pinv(reservAct);    
        
        %outputWeights = outputWeights + 0.1;
               
        
        %testing on testData
        yPredicted = zeros(testLen,outputSize);
        
        for t = 1:testLen %
            u = normData(t+trainLen,:)';
            x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x);
            yPredicted(t) =  outputWeights*[1;u;x];           
        end
        
        yReal = targets(trainLen+1:trainLen + testLen);
        squareErrors = (yReal-yPredicted).^2;
        mseTestTrial(trial) = sum(squareErrors)/length(yPredicted);    
        
        cellWeights(trial,:) = {inputWeights, resWeights, outputWeights};
         
        timing(trial)  = toc;
       
    end
    
    
    
    switch(run)
        
        case 1
            
            save('class_FD001','timing','cellWeights','mseTestTrial');          
            
            
        case 2
            
            
            save('class_FD002','timing','cellWeights','mseTestTrial');
            
        case 3
            
            
            save('class_FD003','timing','cellWeights','mseTestTrial');
            
        case 4
            
            save('class_FD004','timing','cellWeights','mseTestTrial');
            
    end
end
