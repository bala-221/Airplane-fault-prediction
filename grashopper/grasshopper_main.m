
clearvars
close all
clc
rng('shuffle'); %shuffling the random number generator
%This is an impelmenetation of the grasshoper algorithm for fault prrdiction
%network
%
%By: Abubakar Bala
%Topic:
%Started on: 02/04/2019 :DD/MM/YYYY

% I want to run it on all FD001 to FD004

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
    
    
    inputSize = maxSig; %+ size(operatingConditions,2);
    outputSize = 1;
    %trainLen = 21000; % division of test and train data based on 80:20
    %leaky = 0.3;
    %testLen = 9000;
    initLen = 300;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Algorithm parameters
    
    popSize = 20;   %size of the population
    maxGen = 100;  %  atleast for now
    
    
    maxTrial = 10; 
    bestPerIterPerTrail = zeros(maxGen,maxTrial);
    avgPerIterPerTrial =  zeros(maxGen,maxTrial);
    bestCostPerTrial = zeros(maxTrial,1);
    cellWeightsBestPerTrial = cell(maxTrial,3);
    cellBestResExPerTrial = cell(maxTrial,1);
    
    %timing = zeros(maxTrial,1);
    testMSE = zeros(maxTrial,1);
    bestHopperPerTrial = zeros(maxTrial,4);% 3 is the dimension of solution
   
    
    
    cMax = 1;
    cMin = 0.00001;
    
    %population = zeros(popSize,4);
    
    
    resSizeUpperBound = 1500; 
    resSizeLowerBound = 100;
    resConnUpperBound = 0.9;
    resConnLowerBound = 0.1;
    spectRadUpperBound = 2.0;
    spectRadLowerBound = 0.1;
    
   
    
    minFract = 0.2;
    maxFract = 0.7;
    uBound = [resSizeUpperBound, spectRadUpperBound, resConnUpperBound, maxFract]';
    lBound = [resSizeLowerBound, spectRadLowerBound ,resConnLowerBound, minFract]';
    maxDim = size(lBound,1);
    
    
    
    tic;
    parfor trial=1: maxTrial
        %value 3 here because we are saving the resWeghts, inputWeights and
        %ouputWeghts into cells
        leaky = 0.3;
        fitMatrix = zeros(popSize,1);
        cellWeights = cell(popSize,3);
        cellResEx = cell(popSize,1);
        avgPerIter = zeros(maxGen,1);
        bestCostIter = zeros(maxGen,1);
        bestCellWeightsPerIter = cell(maxGen,3);
        bestResExPerIter =  cell(maxGen,1);
        
        %***************************************************************%
        %Generating initial population:
        %We generate initial random populations for the commencement of the
        %Algorithm
        
        reservoirSizes = randi([resSizeLowerBound, resSizeUpperBound],popSize,1);
        spectralRadiis = (spectRadUpperBound-spectRadLowerBound)*rand(popSize,1) + spectRadLowerBound;
        resConnectivitys = (resConnUpperBound-resConnLowerBound)*rand(popSize,1) + resConnLowerBound;
        %leakys = (leakyUpper-leakyLower)*rand(popSize,1) + leakyLower;
        
        dropOutCons = rand(popSize,1)*(maxFract - minFract) + minFract;
        population = [reservoirSizes, spectralRadiis, resConnectivitys, dropOutCons];
        populationNew = zeros(popSize,4);
        
        
        
        for i = 1: popSize
            [fitMatrix(i),cellWeights(i,:),cellResEx{i}] = findCostMSE(population(i,:),normData,targets,inputSize,outputSize,...
                trainLen,initLen,leaky);
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [bestHopperCost, minIndex]  = min(fitMatrix); % find the solution with min cost
        
        bestHopper = population(minIndex,:);
        bestCellWeights = cellWeights(minIndex,:);
        bestResEx = cellResEx{minIndex};
        
        
        newGrassHoppers = zeros(popSize,maxDim);
        
        for gen = 1:maxGen
            
            c = cMax-(gen*((cMax-cMin)/maxGen)); % updating the value of c
            
            %creating new hopper from old ones the main equations in the
            %grasshopper algorithm
            for hopperOne = 1: popSize
                
                temp = population';
                bigSEye = zeros(maxDim,1); %container for summation
                
                for hopperTwo = 1:popSize
                    
                    if hopperOne ~= hopperTwo
                        
                        %finding the distance between two hoppers
                        dist = distance(temp(:,hopperTwo), temp(:,hopperOne));
                        
                        rVector = (temp(:,hopperTwo) - temp(:,hopperOne))/(dist+eps); % xj-xi/dij in Eq. (2.7)
                        
                        xDis = 2 + rem(dist,2); % |xjd - xid| in Eq. (2.7) remember this is the normalization
                        
                        sEye =((uBound - lBound)*c/2)*sFunction(xDis).*rVector; % The first part inside the big bracket in Eq. (2.7)
                        
                        bigSEye = bigSEye + sEye;
                        
                    end
                    
                end
                
                xNew = c * bigSEye' + (bestHopper); % Eq. (2.7) in the paper
                
                %Check for bound violations
                
                % fix the out of bounds
                lowerCut = xNew' < lBound; % find lowerBound violations
                upperCut = xNew' > uBound; % find upperBound violations
                
                cut = lowerCut + upperCut; %Do if only there are violations
                if sum(cut) >= 1
                    xNew(lowerCut) = lBound(lowerCut);
                    xNew(upperCut) = uBound(upperCut);
                end
                
                %round the resSize of xNew
                xNew(1) = round(xNew(1));
                
                %Find the cost of new solution
                
                [fitMatrix(hopperOne),cellWeights(hopperOne,:),cellResEx{hopperOne}] = findCostMSE(xNew,normData,targets,inputSize,outputSize,...
                trainLen,initLen,leaky);
                
                
                
                populationNew(hopperOne,:) = xNew;
                
            end
            
            [newMin,minIndex] =  min(fitMatrix); % find the solution with min cost to replace the best hopper if it is better
            
            population = populationNew;
            
            if newMin < bestHopperCost
                bestHopper = population(minIndex,:);
                bestHopperCost = newMin;
                bestCellWeights = cellWeights(minIndex,:);
                bestResEx = cellResEx{minIndex};
            end
            
            %saving items per iteration
            avgPerIter(gen) = mean(fitMatrix);
            bestCostIter(gen)=  bestHopperCost;
            bestCellWeightsPerIter(gen,:) = bestCellWeights;
            bestResExPerIter{gen} =  bestResEx ;
            
        end
        
        %Saving items per trial
        inputWeights = bestCellWeights{1};
        resWeights = bestCellWeights{2};
        outputWeights= bestCellWeights{3};
        leaky = bestHopper(4);
        
        testMSE(trial) = findTestMSE(inputWeights,resWeights,outputWeights,bestResEx,normData,targets,outputSize,trainLen,testLen,leaky,inputSize,initLen);
  
        cellWeightsBestPerTrial(trial,:) = bestCellWeights;
        cellBestResExPerTrial{trial} = bestResEx;
        bestPerIterPerTrail(:,trial) = bestCostIter;
        bestCostPerTrial(trial) = bestHopperCost;
        avgPerIterPerTrial(:,trial)  =  avgPerIter;
        bestHopperPerTrial(trial,:) = bestHopper ;
    end
    timing = toc;
    
    switch(run)
        
        case 1
            
            save('gHopFD001Cost_rev','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 2
            
            
            save('gHopFD002Cost_rev','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 3
            
            
            save('gHopFD003Cost_rev','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 4
            
            
            save('gHopFD004Cost_rev','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
    end
end










