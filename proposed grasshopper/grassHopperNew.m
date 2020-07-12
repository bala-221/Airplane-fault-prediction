
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

for run = 1:4
    
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
    initLen = 300;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Algorithm parameters
    
    popSize = 20; % %size of the population
    maxGen = 100; %  atleast for now
    
    
    maxTrial = 10;
    bestPerIterPerTrail = zeros(maxGen,maxTrial);
    avgPerIterPerTrial =  zeros(maxGen,maxTrial);
    bestCostPerTrial = zeros(maxTrial,1);
    cellWeightsBestPerTrial = cell(maxTrial,3);
    cellBestResExPerTrial = cell(maxTrial,1);
    
    %timing = zeros(maxTrial,1);
    testMSE = zeros(maxTrial,1);
    bestHopperPerTrial = zeros(maxTrial,20);% 3 is the dimension of solution
    
    
    
    
    
    
    
    leakyMat = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,.9,1]';
    resMat = [100, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500]';
    %resMat = [15, 15, 15,15, 15, 15, 15, 15, 15, 15]';
    spectMat = [0.2, 0.4, 0.6,0.8,1.0,1.2,1.4,1.6, 1.8, 2.0]';
    resConMat = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]';
    regMat = [1E-9,1E-8,1E-7,1E-6,1E-5,1E-4,1E-3,1E-2,1E-1,1]';
    inScaleMat = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6, 1.8,2.0]';
    
    maxDim = 20;
    A = 1;
    goldenRatio = (1+sqrt(5))/2;
    
    
    
    
    uBound = 10;
    lBound = 1;
    
    
    
    tic;
    parfor trial=1: maxTrial
        %value 3 here because we are saving the resWeghts, inputWeights and
        %ouputWeghts into cells
        numChildren = popSize/2;
        fitMatrix = zeros(popSize,1);
        fitMatrixNew = zeros(numChildren,1);
        cellWeights = cell(popSize,3);
        cellWeightsNew = cell(numChildren,3);
        cellResEx = cell(popSize,1);
        cellResExNew = cell(numChildren,1);
        avgPerIter = zeros(maxGen,1);
        bestCostIter = zeros(maxGen,1);
        bestCellWeightsPerIter = cell(maxGen,3);
        bestResExPerIter =  cell(maxGen,1);
        
        %***************************************************************%
        %Generating initial population:
        %We generate initial random populations for the commencement of the
        %Algorithm
        
        population = randi(10,popSize,maxDim);
        populationNew = zeros(numChildren ,maxDim);
        
        for i = 1: popSize
            %[fitMatrix(i),cellWeights(i,:),cellResEx{i}] = findCostMyIdea(population(i,:),normData,targets,inputSize,outputSize,...
            %trainLen,initLen,leakyMat,resMat,spectMat,resConMat,regMat);
            [fitMatrix(i),cellWeights(i,:),cellResEx{i}] = findCost(population(i,:),normData,targets,trainLen,initLen,...
                inputSize,outputSize,leakyMat,resMat,spectMat,resConMat,regMat,inScaleMat);
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [bestHopperCost, minIndex]  = min(fitMatrix); % find the solution with min cost
        
        bestHopper = population(minIndex,:);
        bestCellWeights = cellWeights(minIndex,:);
        bestResEx = cellResEx{minIndex};
        
        newGrassHoppers = zeros(popSize,maxDim);
        
        for gen = 1:maxGen
            
            alpha = A/(gen^(1/2)); %test with various numbers
            
            %creating new hopper from old ones the main equations in the
            %grasshopper algorithm
            
            %dividing the population into two groups
            percentileFit = prctile(fitMatrix,25);
            lessPercentile = fitMatrix <=  percentileFit;
            greaterPercentile = ~lessPercentile;
            
            series = 1:popSize;
            indexesLess = series(lessPercentile);
            indexesGreater = series(greaterPercentile);
            
            bestPop = population(lessPercentile,:);
            worstPop =  population(greaterPercentile,:);
           
            for j = 1: numChildren              
                
                        
                        %selecting hopperOne or two (just like selection in GA) based on rand
                        if rand < 0.7
                             randomIndex = randi(size(bestPop,1));
                             hopperOne = indexesLess(randomIndex);                           
                             firstHopper = population(hopperOne,:);
                        else
                            randomIndex = randi(size(worstPop,1));
                            hopperOne = indexesGreater(randomIndex);
                            firstHopper = population(hopperOne,:);
                        end                        
                        
                        if rand < 0.3 %select from the best                           
                            randomIndex = randi(size(bestPop,1));
                            hopperTwo = indexesLess(randomIndex);
                            secondHopper = population(hopperTwo,:);                           
                        else %select from worst
                            randomIndex = randi(size(worstPop,1));
                            hopperTwo = indexesGreater(randomIndex);
                            secondHopper = population(hopperTwo,:);          
                            
                        end
                        
                        
                                                
                        xNew = zeros(1,maxDim);
                        
                        diff = firstHopper - secondHopper;
                        absDiff = abs(diff);
                        lessThree = absDiff <= 3;
                        greaterSeven = absDiff >= 7;
                        %simulate attraction and repulsion
                        
                        %attraction
                        sumGreat = sum(greaterSeven);
                        if sumGreat >= 1
                            if fitMatrix(hopperOne) < fitMatrix(hopperTwo)
                                diff = firstHopper - secondHopper;
                                ratio = (diff/goldenRatio);
                                presentHopper = population(hopperTwo,:);
                                xNew(greaterSeven) = ratio(greaterSeven) + presentHopper(greaterSeven);
                            else
                                diff = secondHopper - firstHopper;
                                ratio = (diff/goldenRatio);
                                presentHopper = population(hopperOne,:);
                                xNew(greaterSeven) = ratio(greaterSeven) + presentHopper(greaterSeven);
                            end
                        end
                        
                        %repulsion
                        summLess = sum(lessThree);
                        if summLess >= 1
                            value = 0.2*(1-rand(1,maxDim)).^-alpha;
                            idea = randi(2);
                            if idea==1
                                presentHopper = population(hopperOne,:);
                                xNew(lessThree) = value(lessThree) + presentHopper(lessThree);
                            else
                                presentHopper = population(hopperTwo,:);
                                xNew(lessThree) = value(lessThree) +  presentHopper(lessThree);
                            end
                        end
                        
                        %others genes                        
                        otherLogic = ~(greaterSeven | lessThree);                        
                        idea = randi(2);
                        if idea==1
                            presentHopper = population(hopperOne,:);
                            xNew(otherLogic ) = presentHopper(otherLogic);
                        else
                            presentHopper = population(hopperTwo,:);
                            xNew(otherLogic) = presentHopper(otherLogic );
                        end                        
                        
                        
                        %Check for bound violations
                        
                        xNew = round(xNew);
                        % fix the out of bounds
                        lowerCut = xNew' < lBound; % find lowerBound violations
                        upperCut = xNew' > uBound; % find upperBound violations
                        cut = lowerCut + upperCut; %Do if only there are violations
                        if sum(cut) >= 1
                            xNew(lowerCut) = 1;
                            xNew(upperCut) = 10;
                            %xNew(lowerCut) = randi(10,1,sum(lowerCut));
                            %xNew(upperCut) = randi(10,1, sum(upperCut));
                        end
                        %Find the cost of new solution
                        [fitMatrixNew(j),cellWeightsNew(j,:),cellResExNew{j}] = findCost(xNew,normData,targets,trainLen,...
                            initLen,inputSize,outputSize,leakyMat,resMat,spectMat,resConMat,regMat,inScaleMat);
                        populationNew(j,:) = xNew;
                       
                    %end
                %end
            end
            
            %[newMin,minIndex] =  min(fitMatrix); % find the solution with min cost to replace the best hopper if it is better
            
            
            
            
            mergedPopulation = [population;populationNew];
            mergedFitness = [fitMatrix; fitMatrixNew];
            mergedCellWeights = [cellWeights;cellWeightsNew];
            mergedCellResEx = [cellResEx;cellResExNew];
            
            %percentileFit = prctile(mergedFitness,25);
            %lessPercentile = mergedFitness <=  percentileFit;
            %greaterPercentile = ~lessPercentile;
            
            [sortedMergeredFit, indexOfSort] = sort(mergedFitness);
            
            %numAvailable = sum(lessPercentile);
            numAtTop = round(.75 * popSize);
            numOthers = popSize - numAtTop;       
            indexOfBestToChoose =  indexOfSort(1:numAtTop);
            
            indexOfOthersToChoose = indexOfSort(numAtTop+1:end);
            perm = randperm(length(indexOfOthersToChoose),numOthers);
            finalIndexOfOthersToChoose = indexOfOthersToChoose(perm);
            
            
            
            fitMatrix =  [mergedFitness(indexOfBestToChoose);mergedFitness(finalIndexOfOthersToChoose)];
            population = [mergedPopulation(indexOfBestToChoose,:);mergedPopulation(finalIndexOfOthersToChoose,:)];
            cellWeights = [mergedCellWeights(indexOfBestToChoose,:);mergedCellWeights(finalIndexOfOthersToChoose,:)];
            cellResEx = [mergedCellResEx(indexOfBestToChoose);mergedCellResEx(finalIndexOfOthersToChoose)];
            
           
            
            bestHopper = population(1,:);
            bestHopperCost =  fitMatrix(1);
            bestCellWeights = cellWeights(1,:);
            bestResEx = cellResEx{1};
            
           
            avgPerIter(gen) = mean(fitMatrix);
            bestCostIter(gen)=  bestHopperCost;
            bestCellWeightsPerIter(gen,:) = bestCellWeights;
            bestResExPerIter{gen} =  bestResEx ;
        end
        
        %Saving items per trial
        inputWeights = bestCellWeights{1};
        resWeights = bestCellWeights{2};
        outputWeights= bestCellWeights{3};
        
        
        
        
        
        
        testMSE(trial)  = findTestMSE(bestHopper,normData,targets,outputSize,trainLen,testLen,inputSize,initLen,...
            leakyMat,resMat,inputWeights, resWeights,outputWeights,bestResEx);
        
       
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
            
            save('gHopFD001','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial');
            
        case 2
            
            
            save('gHopFD002','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial');
            
        case 3
            
            
            save('gHopFD003','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial');
            
        case 4
            
            
            save('gHopFD004','timing','testMSE','bestPerIterPerTrail','bestCostPerTrial','avgPerIterPerTrial','bestHopperPerTrial');
            
    end
end










