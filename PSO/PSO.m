%Reimplementation of the paper titled: PSO-based analysis of echo state
%network parameters for time series forcasting Chouikhi et al. 2017
%by: Abubakar Bala PhD student
%started on 1st September, 2018, with the name of Allah
%The paper optimizes part of the input weights, reservoir weights and
%feedback weights. PSO is done on training data and tested on fresh test
%data

clearvars
close all
clc
rng('shuffle');

% I want to run it on all FD001 to FD004

for run = 1: 4
    
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
    %targets = (2*(targets - min(targets))/(max(targets)-min(targets))) -1;
    
    
     normData = zeros(size(rawData,1),maxSig);
    for sig = 1: maxSig
        column = rawData(:,sig);
        %newColumn = (2*(column - min(column))/(max(column) - min(column))) -1;
        normData(:,sig) = rescale(column,-1,1);       
    end
    
    
    
    
    %smoothing and removing noise from data
    
    normData = smoothdata(normData,'gaussian');
    
    
    resSize = 1500;
    inSize = maxSig; %+ size(operatingConditions,2);
    outSize = 1;
    resConn = 0.5;
    %trainLen = 2100; % division of test and train data based on 70:30
    leaky = 0.3;
    %testLen = 900;
    initLen = 300;
    fractInputWeights = 0.5;
    fractResWeights = 0.5;
    %fractFeebackWeights = 0.5;
    %resWeights = sprand(resSize,resSize,resConn);
    
  
    maxIter = 100;
    maxTrial = 2;%changed
    numPart = 20; %number of particles
    
    
    lowerBound = -1;
    upperBound = 1;
    
    
    
    bestPerIterPerTrail = zeros(maxIter,maxTrial);
    avgPerIterPerTrial =  zeros(maxIter,maxTrial);
    cellSolutionMatrixPerTrial = cell(maxTrial,1);
    cellRanSelectInputWgtsPerTrial = cell(maxTrial,1);
    cellRanSelectResWgtsPerTrial = cell(maxTrial,1);
    cellWeightsBestPerTrial = cell(maxTrial,3);
    cellBestResExPerTrial = cell(maxTrial,1);
    %timing = zeros(maxTrial,1);
    testMSE = zeros(maxTrial,1);
    bestCostPerTrial = zeros(maxTrial,1);
    cellBestNestPerTrial = cell(maxTrial,1);
   
    
    tic;
    parfor trial = 1: maxTrial
        
        costMatrix = zeros(numPart,1);
        cellWeights = cell(numPart,3);
        cellResEx = cell(numPart,1);
        newCellWeights = cell(numPart,3);
        newCellResEx = cell(numPart,1);
        bestPerIter =  zeros(maxIter,1);
        avgPerIter = zeros(maxIter,1);
        
        
        
        resWeights =  sprand(resSize,resSize,resConn);%-0.5;
        
        resWghtsMask = (resWeights~=0);
        
        resWeights(resWghtsMask) = (resWeights(resWghtsMask)*2 -1); %#ok<SPRIX>
        
        %the +1 here is for the bias unit
        inputWeights = rand(resSize,1+inSize)*2-1 ;       
        
        %feedbackWeights = sprand(resSize,outputSize,0.1);
        
        
        %how to select the initial
        %fraction of weights selected for optimization
        
        
        %selecting part of the inputWeights for optimization
        randSelectedInputWeights = randperm(numel(inputWeights),round(fractInputWeights*numel(inputWeights)));
        %randInputWeights =  inputWeights(randSelectedInputWeights,2);
        
        
        %selecting part of the reservoir Weights for optimization
        nonZeroInidices = find(resWeights);
        randSelectedResWeights = randperm(nnz(resWeights),round(fractResWeights*nnz(resWeights)));
        randSelectedResWeights = nonZeroInidices(randSelectedResWeights);
        %randResWeights = resWeights(randSelectedResWeights);
        
        %     %selecting part of the feedback Weights for optimization
        %     nonZeroInidices = find(feedbackWeights);
        %     randSelectedFbWeights = randperm(nnz(feedbackWeights),round(fractFeebackWeights*nnz(feedbackWeights)));
        %     randSelectedfbWeights = nonZeroInidices(randSelectedFbWeights);
        %     %randFbWeights = feedbackWeights(randSelectedfbWeights);
        %
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Algorthm
        %initial population
        
        %dimension =  length(randInputWeights) + length(randResWeights) + length(randFbWeights);
        dimSize =  length(randSelectedInputWeights) + length(randSelectedResWeights);
        %creating a initial population between the upperBound and lowerBound
        solutionMatrix = rand(numPart,dimSize)*(upperBound-lowerBound) + lowerBound;
        %solutionMatrix = rand(numPart,dimension);
        
        
        %finding the cost of each solution
        
        for i=1: numPart
            solution = solutionMatrix(i,:);
            [costMatrix(i), cellWeights(i,:), cellResEx{i}] = findCostMSE(solution,normData,targets,randSelectedInputWeights,randSelectedResWeights,...
                inputWeights,resWeights,inSize,resSize,outSize,trainLen,initLen,leaky);
        end
        
        %PSO
        w = 0.9;
        c1 = 0.12;
        c2 = 2.2;
        %initialize the velocity
        
        velocityMatrix = rand(numPart,dimSize)*2-1;
        
        [cost,index]= min(costMatrix);
        partBestMatrix =  solutionMatrix;
        
        partBestCosts = costMatrix;
        
        globalBestPart = solutionMatrix(index,:);
        
        globalBestCost = cost;
        
        bestCellWeights = cellWeights(index,:);
        bestResEx  = cellResEx{index};
        
        for iter = 1: maxIter
            % updating the velocity and checking for bound violation
            
            
            
            %update velocity of each Particle
            for part = 1: numPart
                
                velocityMatrix(part,:) = w*velocityMatrix(part,:) ...
                    + c1*rand(1,dimSize).*(partBestMatrix(part,:)-solutionMatrix(part,:))+...
                    c2.*rand(1,dimSize).*(globalBestPart-solutionMatrix(part,:));
                
                
                % Adding the velocity to the particle
                solutionMatrix(part,:) = solutionMatrix(part,:) +  velocityMatrix(part,:);
                
                tinySol = solutionMatrix(part,:) ;
                
                % Check for bound violations
                lowerCut = tinySol  < lowerBound; % find lowerBound violations
                upperCut = tinySol  > upperBound; % find upperBound violations
                
                tinySol(lowerCut) = lowerBound;
                tinySol(upperCut) = upperBound;
                
                %Find the cost of the new solution
                
                [newfitMatrix,cellNewWeights,cellNewResEx] = findCostMSE(tinySol,normData,targets,randSelectedInputWeights,randSelectedResWeights,...
    inputWeights,resWeights,inSize,resSize,outSize,trainLen,initLen,leaky);
                
                
                % Should there be a replacement in the particle bests?
                
                if newfitMatrix <= partBestCosts(part)
                    partBestCosts(part) = newfitMatrix;
                    partBestMatrix(part,:) = tinySol;
                    cellWeights(part,:) = cellNewWeights;
                    cellResEx{part} = cellNewResEx;
                end
                
            
                
            end
            
                       
            
            %update the global Best
            
            [cost,index] = min(partBestCosts);
            
            if cost < globalBestCost
                globalBestCost = cost;
                globalBestPart = partBestMatrix(index,:);
                bestCellWeights = cellWeights(index,:);
                bestResEx = cellResEx{index};
            end
            
            
            
            bestPerIter(iter) =  globalBestCost;
            avgPerIter(iter) = mean(partBestCosts);
            
        end
        
        %finding the test performance
        
        
        inputWeights = bestCellWeights{1};
        resWeights = bestCellWeights{2};
        outputWeights = bestCellWeights{3};
        
        testMSE(trial) = findTestMsePso(outputWeights,resWeights,inputWeights,normData,targets,outSize,trainLen,testLen,leaky,bestResEx);
                
        cellBestNestPerTrial{trial} = globalBestPart;
        bestPerIterPerTrail(:,trial) = bestPerIter;
        avgPerIterPerTrial(:,trial)  =  avgPerIter;
        bestCostPerTrial(trial) = globalBestCost;
        cellSolutionMatrixPerTrial{trial,1} = solutionMatrix;
        cellRanSelectInputWgtsPerTrial{trial,1} = randSelectedInputWeights;
        cellRanSelectResWgtsPerTrial{trial,1} = randSelectedResWeights;
        cellWeightsBestPerTrial(trial,:) = bestCellWeights;
        cellBestResExPerTrial{trial} = bestResEx;
    end
    
    timing = toc;
    
    switch(run)
        
        case 1
            
            save('PSO_FD001_time','timing','cellBestNestPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 2
            
            
            save('PSO_FD002_time','timing','cellBestNestPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
        
        case 3
            
            
            save('PSO_FD003_time','timing','cellBestNestPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
        
        case 4
            
            
            save('PSO_FD004_time','timing','cellBestNestPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
    end
    
end


