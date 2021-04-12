
clearvars
close all
clc
rng('shuffle'); %shuffling the random number generator
%This is my Thesis topic on cuckoo search for optimizing the echo state
%network
%
%By: Abubakar Bala
%Topic:
%Started on: 22/02/2018 :DD/MM/YYYY
% for now, we are optimizing only the number of reservoir (N), Spectral
% Radii(Sr) and reservoirConnectivity, and scaling of reservoir weights and
% input weights
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
    leaky = 0.3;
   
    initLen = 300;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Algorithm parameters
    
    popSize = 20;  %size of the population
    A = 1; % maximum levy flight step size;
    goldenRatio = (1+sqrt(5))/2;
    maxGen = 100; %atleast for now
    
    
    
    maxTrial = 10; 
    bestPerIterPerTrail = zeros(maxGen,maxTrial);
    avgPerIterPerTrial =  zeros(maxGen,maxTrial);
    bestTrainPerTrial = zeros(maxTrial,1);
    cellWeightsBestPerTrial = cell(maxTrial,3);
    cellBestResExPerTrial = cell(maxTrial,1);
    bestNestPerTrial = zeros(maxTrial,4); % 4 here is the diemnsion size of the nest
    
    %timing = zeros(maxTrial,1);
    testMSE = zeros(maxTrial,1);
    minFract = 0.2;
    maxFract = 0.7;
    
    percentageBottumNests = 0.75;
    numBottomNests = floor(percentageBottumNests*popSize);
    numTopNests = popSize - numBottomNests;
    population = zeros(popSize,4);
    
    resSizeUpperBound = 1500;
    resSizeLowerBound = 100;
    resConnUpperBound = 0.9;
    resConnLowerBound = 0.1;
    spectRadUpperBound = 2.0;
    spectRadLowerBound = 0.1;
    
    
    uBound = [resSizeUpperBound, spectRadUpperBound, resConnUpperBound, maxFract]';
    lBound = [resSizeLowerBound, spectRadLowerBound ,resConnLowerBound, minFract]';
    
    
    
    tic;
  
    parfor trial=1: maxTrial
       
        costMatrix = zeros(popSize,1);
        cellWeights = cell(popSize,3);
        cellResEx = cell(popSize,1);
        avgPerIter = zeros(maxGen,1);
        bestCostIteration = zeros(maxGen,1);
        
        %***************************************************************%
        %Generating initial population:
        %We generate initial random populations for the commencement of the
        %Algorithm
        
        reservoirSizes = randi([resSizeLowerBound, resSizeUpperBound],popSize,1);
        spectralRadiis = (spectRadUpperBound-spectRadLowerBound)*rand(popSize,1) + spectRadLowerBound;
        resConnectivitys = (resConnUpperBound-resConnLowerBound)*rand(popSize,1) + resConnLowerBound;
        dropOutCons = rand(popSize,1)*(maxFract - minFract) + minFract;
        population = [reservoirSizes, spectralRadiis, resConnectivitys, dropOutCons];
        
        
        
        for i = 1: popSize
            [costMatrix(i),cellWeights(i,:),cellResEx{i}] = findCostMSE(population(i,:),normData,targets,inputSize,outputSize,...
                trainLen,initLen,leaky);
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        [costMatrix, sortIndex]  = sort(costMatrix);
        
        population = population(sortIndex,:);
        
        cellWeights = cellWeights(sortIndex,:);
        cellResEx = cellResEx(sortIndex,:);
        
        for gen = 1:maxGen
            
            
            topNests  = population(1:numTopNests,:);
            
            bottomNests = population(numTopNests+1:end,:);
            
            
            newBottomNests = zeros(numBottomNests,4);
            
            costOfNewBottomNests = zeros(numBottomNests,1);
            cellWeightsBottomNest = cell(numBottomNests,3);
            cellBotResEx = cell(numBottomNests,1);
            
            for botNest = 1:numBottomNests
                
                solution = bottomNests(botNest,:);
                
                % generating levy flight
                
                alpha = A/(gen^(1/7)); %test with various numbers
                
                levyFlightForResSize = 5*round((1-rand)^-alpha) ;  %levy flight functions maximum ...
                %is 5 minimum is 2 %atleast for now
                
                levyFlightForSpectral = 0.0075*(1-rand)^-alpha; % 5 percent of max
                
                levyFlightResConnectivity = 0.0025*(1-rand)^-alpha;
                
                dropOutConLevy = minFract*(1-rand)^-alpha  ;
                
                
                
                solutionNew = solution + sign(rand - 0.5)*[levyFlightForResSize,levyFlightForSpectral,levyFlightResConnectivity,dropOutConLevy];
                
                
                
                % fix the out of bounds
                lowerCut =  solutionNew' < lBound; % find lowerBound violations
                upperCut =  solutionNew' > uBound; % find upperBound violations
                
                cut = lowerCut + upperCut; %Do if only there are violations
                
                if sum(cut) >= 1
                    solutionNew(lowerCut) = lBound(lowerCut);
                    solutionNew(upperCut) = uBound(upperCut);
                end
                
                
                newBottomNests(botNest,:) = solutionNew;
                [costOfNewBottomNests(botNest),cellWeightsBottomNest(botNest,:), cellBotResEx{botNest}] = findCostMSE(solutionNew,normData,targets,inputSize,outputSize,...
                    trainLen,initLen,leaky);
                
            end
            
            costMatrix = [costMatrix(1:(end-numBottomNests));costOfNewBottomNests];
            
            population = [topNests;newBottomNests]  ;  %the new total solution after adding newly created bottom nests
            
            cellWeights(numTopNests+1:end,:) = cellWeightsBottomNest;
            
            cellResEx(numTopNests+1:end,:) = cellBotResEx;
            
            
            for topNest = 1: numTopNests
                
                %pick two random top nests
                
                randomTopOne = randi(numTopNests,1);
                
                %if they are equal? perform Levy flight
                if randomTopOne == topNest
                    
                    
                    solution = topNests(randomTopOne,:);
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    % generating levy flight for number of bins to remove or delete
                    
                    alpha = A/(gen^(1/4)); %test with various numbers
                    
                    levyFlightForResSize = 5*round((1-rand)^-alpha) ;  %levy flight functions maximum ...
                    %is 5 minimum is 2 %atleast for now
                    
                    levyFlightForSpectral = 0.075*(1-rand)^-alpha; % 5 percent of min
                    
                    levyFlightResConnectivity = 0.0025*(1-rand)^-alpha;
                    
                    
                    dropOutConLevy = minFract*(1-rand)^-alpha ;
                    
                    solutionNew = solution + sign(rand - 0.5)*[levyFlightForResSize,levyFlightForSpectral,levyFlightResConnectivity,dropOutConLevy];
                    
                    
                    %check for bound violations
                    
                    % fix the out of bounds
                    lowerCut =  solutionNew' < lBound; % find lowerBound violations
                    upperCut =  solutionNew' > uBound; % find upperBound violations
                    
                    cut = lowerCut + upperCut; %Do if only there are violations
                    
                    if sum(cut) >= 1
                        solutionNew(lowerCut) = lBound(lowerCut);
                        solutionNew(upperCut) = uBound(upperCut);
                    end
                    
                    
                    [costSolutionNew,newCellWeights, newCellResEx] = findCostMSE(solutionNew,normData,targets,inputSize,outputSize,...
                        trainLen,initLen,leaky);
                    
                    %now select a random solution from the entire population and
                    %compare with solutionNew
                    randomIndex = randi(popSize,1);
                    randomSolutionL =   population(randomIndex,:);
                    
                    costRandomSolutionL = costMatrix(randomIndex);
                    
                    if costSolutionNew < costRandomSolutionL
                        population(randomIndex,:) = solutionNew;
                        costMatrix(randomIndex) = costSolutionNew;
                        cellWeights(randomIndex,:) = newCellWeights;
                        cellResEx{randomIndex} = newCellResEx;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                else % we perform crossover of the two solution and pick the best of the children
                    
                    %take dropOutCon either from either of the top nests
                    idea = randi(2);
                    
                    if idea == 1
                        dropOutCon = topNests(randomTopOne,4);
                        
                    else
                        dropOutCon = topNests(topNest,4);
                    end
                    
                    
                    topSolutionOne = topNests(randomTopOne,1:3);
                    topSolutionTwo = topNests(topNest,1:3);
                    
                    
                    
                    dx = abs(topSolutionOne - topSolutionTwo)/goldenRatio;
                    
                    costFirstSolution = costMatrix(randomTopOne);
                    
                    costSecondSolution = costMatrix(topNest);
                    
                    if costFirstSolution > costSecondSolution
                        
                        solutionNew = topSolutionOne + dx;
                        
                    elseif  costSecondSolution > costFirstSolution
                        
                        solutionNew = topSolutionTwo + dx;
                    end
                    
                    
                    %let's round up the resSize.
                    
                    solutionNew(1) = round(solutionNew(1));
                    
                    solutionNewNew = [solutionNew,dropOutCon];
                    
                    %check bound violations
                    
                    lowerCut =  solutionNewNew' < lBound; % find lowerBound violations
                    upperCut =  solutionNewNew' > uBound; % find upperBound violations
                    
                    cut = lowerCut + upperCut; %Do if only there are violations
                    
                    if sum(cut) >= 1
                        solutionNewNew(lowerCut) = lBound(lowerCut);
                        solutionNewNew(upperCut) = uBound(upperCut);
                    end
                    
                    
                    
                    
                    
                    %now select a random solution from the entire population and
                    %compare with offspring generated
                    
                    randomIndex = randi(size(population,1),1);
                    randomSolutionL =   population(randomIndex,:);
                    
                    [costSolutionNew,newCellWeights, newCellResEx] = findCostMSE(solutionNewNew,normData,targets,inputSize,outputSize,...
                        trainLen,initLen,leaky);
                    
                    
                    costRandomSolutionL = costMatrix(randomIndex);
                    
                    if  costSolutionNew <  costRandomSolutionL
                        population(randomIndex,:) = solutionNewNew ;
                        costMatrix(randomIndex) = costSolutionNew;
                        cellWeights(randomIndex,:) = newCellWeights;
                        cellResEx{randomIndex} = newCellResEx;
                    end
                    
                end
            end
            
            
            [costMatrix, sortIndex]  = sort(costMatrix);
            
            population = population(sortIndex,:);
            
            cellWeights = cellWeights(sortIndex,:);
            cellResEx = cellResEx(sortIndex,:);
            avgPerIter(gen) = mean(costMatrix);
            bestCostIteration(gen)= costMatrix(1);
            bestCellWeights = cellWeights(sortIndex(1),:);
            bestResEx = cellResEx{sortIndex(1)};            
            
            
        end
        inputWeights = bestCellWeights{1};
        resWeights = bestCellWeights{2};
        outputWeights= bestCellWeights{3};
        testMSE(trial) = findTestMseCuckoo(inputWeights,resWeights,outputWeights,normData,targets,outputSize,trainLen,testLen,leaky,bestResEx);
        % bestNest = population(1,:);
        
       
        cellWeightsBestPerTrial(trial,:) = bestCellWeights;
        cellBestResExPerTrial{trial} = bestResEx;
        bestPerIterPerTrail(:,trial) = bestCostIteration;
        bestTrainPerTrial(trial) = bestCostIteration(end);
        avgPerIterPerTrial(:,trial)  =  avgPerIter;
        bestNestPerTrial(trial,:) = population(1,:);
        
    end
    
    timing = toc;
    
    
    switch(run)
        
        case 1
            save('cs_FD001','timing','bestNestPerTrial','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');    
            
        case 2
            
            
            save('cs_FD002','timing','bestNestPerTrial','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');   
            
        case 3
            
            
            save('cs_FD003','timing','bestNestPerTrial','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');   
            
        case 4
            
            
            save('cs_FD004','timing','bestNestPerTrial','bestPerIterPerTrail','bestTrainPerTrial','avgPerIterPerTrial','testMSE','cellWeightsBestPerTrial','cellBestResExPerTrial');   
            
    end
end
















