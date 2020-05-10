
%This is an implementation of the differential evolution (DE) algorithm for
%ESN optimisation from the paper: Effective electricity consumption
%forecasting using echo state network improved by diffrential evolution
%algorithm by Wang et al. 2018
%The DE implemneted is the jDE from Brest J, Greiner S, Boškovi´c B,
%Mernik M, Žumer V (2006) Self-adapting control parameters in differential
%evolution: a comparative study on numerical benchmark problems.
%IEEE Trans Evol Comput 10(6):646–657
%otpimizes only three parameters
%By: Abubakar Bala
%Topic:
%Started on: 04/30/2019 :DD/MM/YYYY
clearvars
close all
clc
rng('shuffle'); %shuffling the random number generator

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
    
    inputSize = maxSig; %+ size(operatingConditions,2);
    outputSize = 1;
    leaky = 0.3;
    initLen = 300;
    popSize = 20; 
    maxGen = 100; 
    
    
    
    maxTrial = 10;
    testMSE = zeros(maxTrial,1);
    bestPerIterPerTrail = zeros(maxGen,maxTrial);
    avgPerIterPerTrial =  zeros(maxGen,maxTrial);
    bestTrainPerTrial = zeros(maxTrial,1);
    cellWeightsBestPerTrial = cell(maxTrial,3); % 3 here denote input weights, reservoir weights and ouput weights
    cellBestResExPerTrial = cell(maxTrial,1);
    bestSolPerTrial = zeros(maxTrial,3); % 3 here is the diemnsion size of the nest
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    resSizeUpperBound = 1500;
    resSizeLowerBound = 100;
    resConnUpperBound = 0.9;
    resConnLowerBound = 0.1;
    spectRadUpperBound = 2.0;
    spectRadLowerBound = 0.1;
    
    uBound = [resSizeUpperBound, spectRadUpperBound, resConnUpperBound]';
    lBound = [resSizeLowerBound, spectRadLowerBound ,resConnLowerBound]';
    maxDim = size(lBound,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
     tic;
     parfor trial = 1: maxTrial
        
        %algorithm parameters
        fu = 0.9;
        fl = 0.1;
        fEye = 0.5;
        tauOne = 0.1;
        tauTwo = 0.1;
        cR = 0.9;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        fitMatrix = zeros(popSize,1);
        cellWeights = cell(popSize,3);
        cellResEx = cell(popSize,1);
        avgPerIter = zeros(maxGen,1);
        bestCostIter = zeros(maxGen,1);
        bestCellWeightsPerIter = cell(maxGen,3);
        bestResExPerIter =  cell(maxGen,1);
        
        
        reservoirSizes = randi([resSizeLowerBound, resSizeUpperBound],popSize,1);
        spectralRadiis = (spectRadUpperBound-spectRadLowerBound)*rand(popSize,1) + spectRadLowerBound;
        resConnectivitys = (resConnUpperBound-resConnLowerBound)*rand(popSize,1) + resConnLowerBound;
        population = [reservoirSizes, spectralRadiis, resConnectivitys];
        
        
        for i = 1: popSize
            [fitMatrix(i),cellWeights(i,:),cellResEx{i}] = findCostNoD(population(i,:),normData,targets,inputSize,outputSize,...
                trainLen,initLen,leaky);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %main algorithm
        for gen = 1: maxGen
            
            for i = 1: popSize
                
                %fi update
                
                if rand < tauOne
                    fEye = fl + fu*rand;
                end
                
                %mutation
                %selecting three random (unique) individuals
                %indiOne ~= indiTwo ~= indiThree ~= i
                
                %finding a matrix with all indexes
                value = 1:popSize;
                
                %removing the present i
                value(value==i) = [];
                
                %finding the three unique values for mutation
                randomIndexes = value(randperm(numel(value),3));
                
                indiOne = population(randomIndexes(1),:);
                indiTwo = population(randomIndexes(2),:);
                indiThree = population(randomIndexes(3),:);
                
                
                offspring = indiOne + fEye*(indiTwo - indiThree);
                
                %fix out of bounds
                
                lowerCut = offspring' < lBound; % find lowerBound violations
                upperCut = offspring' > uBound; % find upperBound violations
                
                cut = lowerCut + upperCut; %Do if only there are violations
                if sum(cut) >= 1
                    offspring(lowerCut) = lBound(lowerCut);
                    offspring(upperCut) = uBound(upperCut);
                end
                
                %cr update
                if rand < tauTwo
                    cR = rand;
                end
                
                
                % while loop is to ensure that atlease one of the genes of child is from
                % offsping found from mutation
                lessCross = zeros(1,maxDim);
                crossChild = zeros(1,maxDim);
                while sum(lessCross) == 0
                    
                    %generate random numbers
                    randomCross = rand(1,maxDim);
                    
                    %Check if they are less than the cr
                    lessCross = randomCross <= cR;
                    
                    %if they are take gene from offspring
                    crossChild(lessCross) = offspring(lessCross);
                    
                    %else take from parant
                    parent = population(i,:);
                    crossChild(~lessCross) = parent(~lessCross);
                    
                    
                    
                end
                
                % first round up the # reservoir units in child since it has to
                % be an integer
                crossChild(1) = round(crossChild(1));
                
                %selection one-to-one spawning selection
                %selection only if offspring outperfrom parent
                
                %find the cost of crossChild
                
                [childFitness,childcellWeights,childCellResEx] = findCostNoD(crossChild,normData,targets,inputSize,outputSize,...
                    trainLen,initLen,leaky);
                
                
                
                % if the child is better perform replacement
                if childFitness <= fitMatrix(i)
                    population(i,:) = crossChild;
                    fitMatrix(i)= childFitness;
                    cellWeights(i,:) = childcellWeights;
                    cellResEx{i} = childCellResEx;
                    
                end
                
            end
            
            % save parameters
            [minFitness, index]  = min(fitMatrix);
            
            
            avgPerIter(gen) = mean(fitMatrix);
            bestCostIter(gen)= fitMatrix(index);
            bestCellWeights = cellWeights(index,:);
            bestResEx = cellResEx{index};
            bestCellWeightsPerIter(gen,:) = bestCellWeights;
            
            
        end
        
                inputWeights = bestCellWeights{1};
                resWeights = bestCellWeights{2};
                outputWeights = bestCellWeights{3};

        bestSol  = population(index,:);
        testMSE(trial) = funcTestMSE(inputWeights,resWeights,outputWeights,normData,targets,outputSize,trainLen,testLen,leaky,bestResEx);
        
        %testMSE(trial) = testNawaRMSE(inputWeights,resWeights,outputWeights,bestResEx,normData,targets,outputSize,trainLen,testLen,leaky);
        
        cellWeightsBestPerTrial(trial,:) = bestCellWeights;
        cellBestResExPerTrial{trial} = bestResEx;
        bestPerIterPerTrail(:,trial) = bestCostIter;
        bestTrainPerTrial(trial) = bestCostIter(end);
        avgPerIterPerTrial(:,trial)  =  avgPerIter;
        bestSolPerTrial(trial,:) = bestSol ;
        
        
    end
    
    timing = toc;
    
    switch(run)
        
        case 1
            save('DE_FD001_rev','timing','testMSE','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','bestSolPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 2            
            
            save('DE_FD002_rev','timing','testMSE','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','bestSolPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 3
            
            
            save('DE_FD003_rev','timing','testMSE','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','bestSolPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
        case 4
            
            
            save('DE_FD004_rev','timing','testMSE','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','bestSolPerTrial','cellWeightsBestPerTrial','cellBestResExPerTrial');
            
    end
end
