%From the paper titled: Optimizing the echo state network with a binary particle swarm
%optimization algorithm
%by: Abubakar Bala PhD student
%started on 7th June, 2018
%The otimization is carriedout on the validation set


clearvars
close all
clc
rng('shuffle');


for run = 1: 4
    
    switch(run)
        
         case 1
            fid = fopen('FD001_edited.txt');
            cdata  =  textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', 'HeaderLines',1);
            veryRawData = cell2mat(cdata);
            fclose(fid);    
            
            trainLen = 14000; 
            testLen = 6000; 
            
        case 2
            
            fid = fopen('FD002_edited.txt');
            cdata  =  textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', 'HeaderLines',1);
            veryRawData = cell2mat(cdata);
            fclose(fid);         
           
            trainLen = 35000;
            testLen = 15000;
            
        case 3
            
             fid = fopen('FD003_edited.txt');
            cdata  =  textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', 'HeaderLines',1);
            veryRawData = cell2mat(cdata);
            fclose(fid);    
            trainLen = 14000;
            testLen = 6000;
            
        case 4
            
            fid = fopen('FD004_edited.txt');
            cdata  =  textscan(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', 'HeaderLines',1);
            veryRawData = cell2mat(cdata);
            fclose(fid);    
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
    
    validLen =  round(trainLen*0.3);
    resSize = 1500;
    inputSize = maxSig; %+ size(operatingConditions,2);
    outputSize = 1;
    resConn = 0.5;
    
    leaky = 0.3;
    spectralRad = 1;
    initLen = 300;
    
    
    maxIter = 100; 
    maxTrial = 10;
    numPart = 20;
    
    
    
    cellBestPartPerTrial = cell(maxTrial,1);
    bestPerIterPerTrail = zeros(maxIter,maxTrial);
    bestCostPerTrial = zeros(maxTrial,1);
    avgPerIterPerTrial =  zeros(maxIter,maxTrial);
    testMSE = zeros(maxTrial,1);
    cellWeightsBestPerTrial = cell(maxTrial,3);
    cellBestResExPerTrial = cell(maxTrial,1);
    cellSolutionMatrixPerTrial = cell(maxTrial,1);
    
    
    
    
    tic;
    
    parfor trial=1: maxTrial  
        
        
        
        cellWeights = cell(numPart,3);
        cellResEx = cell(numPart,1);
        bestPerIter = zeros(maxIter,1);
        avgPerIter = zeros(maxIter,1);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        resWeights =  sprand(resSize,resSize,resConn);%-0.5;        
        resWghtsMask = (resWeights~=0);        
        resWeights(resWghtsMask) = (resWeights(resWghtsMask)*2-1);         
      
        rhoW = abs(eigs(resWeights ,1));      
                
        value = 25;
        while  isnan(rhoW)
            rhoW = abs(eigs(resWeights ,1,'LM','SubspaceDimension',value));
            value = value + 3;
        end   
        
        resWeights  = resWeights .* (spectralRad/rhoW);
        %the +1 here is for the bias unit
        inputWeights = rand(resSize,1+inputSize)*2-1 ;
        %feedbackWeights = sprand(resSize,outputSize,0.1);        
        %Training
        newTrainLen = round(trainLen*0.7);
        
        reservAct =  findActivations(inputWeights,resWeights,normData,newTrainLen,initLen,...
            leaky,inputSize,resSize);      
       
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
       
        
        
        c1 = 2;
        c2 = 2;
        w_min = 0.1;
        w_max = 0.6;
        dimSize = resSize+inputSize+1; % particle dimension size
        
        %making the velicty range [-1,1]
       
        velocityMatrix = rand(numPart,dimSize)*2 - 1;
        %Generating random Solutions binary:
        solutionMatrix = randi([0 1],numPart,resSize+inputSize+1);
        
        
        
        %find the cost of each solution
        costMatrix = zeros(numPart,1);
        
        for i=1:numPart
            particle = solutionMatrix(i,:);
            [costMatrix(i),cellWeights(i,:), cellResEx{i}] = findCostMSE(normData,particle,reservAct,initLen,trainLen,...
                validLen,targets,outputSize,x,inputWeights,resWeights);
        end
        
        particleBestMatrix = solutionMatrix;
        particleBestCost = costMatrix;
        [cost,index] = min(costMatrix);
        
        
        globalBestCost = cost;
        globalBestPart = solutionMatrix(index,:);
        
        bestCellWeights = cellWeights(index,:);
        bestResEx  = cellResEx{index};
        
        
        
        for iter = 1: maxIter
            
            w = w_max+(((w_min-w_max)*(iter-1))/(maxIter-1));
            
            %update velocity of Particle
            for part = 1: numPart
                
                              
                velocityMatrix(part,:) = w*velocityMatrix(part,:)...
                    + c1*rand(1,dimSize).*(particleBestMatrix(part,:)-solutionMatrix(part,:))+...
                    c2*rand(1,dimSize).*(globalBestPart-solutionMatrix(part,:));
              
                %make it binary and update the bits in every particle
                pro_velocity = abs(2*(logsig(velocityMatrix(part,:))-0.5));               
                
                %Making changes to the actual solution
                solutionMatrixSmall = solutionMatrix(part,:);                
                logicalIndexes = rand(1,dimSize) < pro_velocity;                
                
                solutionMatrixSmall(logicalIndexes) = xor(solutionMatrixSmall(logicalIndexes),1);               
                
                
                %findng the new cost
                particle = solutionMatrixSmall;
                [newfitMatrix,cellNewWeights,cellNewResEx] = findCostMSE(normData,particle,reservAct,initLen,trainLen,...
                    validLen,targets,outputSize,bestResEx,inputWeights,resWeights);
                
                
                %Should there be a replacement
                
                if newfitMatrix <= particleBestCost(part)
                    particleBestCost(part) = newfitMatrix;
                    particleBestMatrix(part,:) =  particle;
                    cellWeights(part,:) = cellNewWeights;
                    cellResEx{part} = cellNewResEx;
                end
                
                
                
                
                
            end
            
            
            %update the global Best
            
            [cost,index] = min(particleBestCost);
            
            if cost < globalBestCost
                globalBestCost = cost;
                globalBestPart = particleBestMatrix(index,:);
                bestCellWeights = cellWeights(index,:);
                bestResEx = cellResEx{index};
            end
            
            
            
            bestPerIter(iter) =  globalBestCost;
            avgPerIter(iter) = mean(particleBestCost);
            
        end
        
        
        
        inputWeights = bestCellWeights{1};
        resWeights = bestCellWeights{2};
        outputWeights = bestCellWeights{3};
        
        testMSE(trial) = findTestBpsoMse(normData,outputWeights,trainLen,testLen,targets,...
            outputSize,inputWeights,resWeights,bestResEx);
        
        
        cellBestPartPerTrial{trial} = globalBestPart;
        bestPerIterPerTrail(:,trial) = bestPerIter;
        avgPerIterPerTrial(:,trial)  =  avgPerIter;
        bestCostPerTrial(trial) = globalBestCost;
        cellSolutionMatrixPerTrial{trial,1} = solutionMatrix;
        cellWeightsBestPerTrial(trial,:) = bestCellWeights;
        cellBestResExPerTrial{trial} = bestResEx;
        
        
    end
    
    timing = toc;
    
    switch(run)
        
        case 1
            
            save('BPSO_FD001','timing','cellBestPartPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 2
            
            
            save('BPSO_FD002','timing','cellBestPartPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 3
            
            
            save('BPSO_FD003','timing','cellBestPartPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 4
            
            
            save('BPSO_FD004','timing','cellBestPartPerTrial','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
    end
    
end




