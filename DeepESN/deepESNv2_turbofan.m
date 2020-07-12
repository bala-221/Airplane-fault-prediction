
clearvars
close all
clc
rng('shuffle'); %shuffling the random number generator

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
      
    
    normData = zeros(size(rawData,1),maxSig);
    
     
    for sig = 1: maxSig
        column = rawData(:,sig);
        %newColumn = (2*(column - min(column))/(max(column) - min(column))) -1;
        normData(:,sig) = rescale(column,-1,1);       
    end
    
    %smoothing and removing noise from data
    
    normData = smoothdata(normData,'gaussian');
    
 
    maxResSize = 1500;
    resConn = 0.5;
    inputSize = maxSig; %+ size(operatingConditions,2);
    outputSize = 1;
    leaky = 0.3;
    washout = 300;    
    reg = 1e-8;
       
    numLayers = 3;  
    resSize = maxResSize/numLayers;
    
    bias = 1;    
    
    
    
    maxTrial = 10;
    cost = zeros(maxTrial,1);
    mseTestTrial = zeros(maxTrial,1);
    cellWeights = cell(maxTrial,3);
    timing = zeros(maxTrial,1);
    spectralRadius = 1;
    resCon = 0.5; % reservoir connectivity
    
    
    
    
for trial = 1: maxTrial
    tic;
    Win = rand(resSize,1+inputSize)*2-1; %range is [-1, 1]
    % generate the ESN reservoir
    
    %Creating different reservoir weights for each layer
    cellResWeights = cell(numLayers,1);
    
    for i = 1: numLayers        
       
        % sparse W:        
        W = sprand(resSize,resSize, resCon);
        W_mask = (W~=0);
        W(W_mask) = (W(W_mask)*2 -1); 
        % normalizing and setting spectral radius
        disp 'Computing spectral radius...';
        opt.disp = 0;
        rhoW = abs(eigs(W,1,'LM',opt));
        
        value = 25;
        while  isnan(rhoW)
            rhoW = abs(eigs(W ,1,'LM','SubspaceDimension',value));
            value = value + 3;
        end   
        
        disp 'done.'
        W = W .* (spectralRadius  /rhoW);        
        cellResWeights{i} = W;
    end
    
    % Different interlayer weights
    cellInterLayer = cell(numLayers,1);
    for i = 2: numLayers
        cellInterLayer{i} = rand(resSize,1+resSize)*2-1;
    end
    
    
    cellPerLayer = cell(numLayers,1); % cell that hold all states for each layer
    %as a matrix in a cell array
    
    %Generating the initial zeros for each state per training points
    cellPerLayer(1:numLayers) = {zeros(resSize, trainLen)};
    
    cellTrackLayerStates = cell(numLayers,1); % This cell is responsible for holding the values
    %of each states for each layers, it gets overwritten after each time stamp
    cellTrackLayerStates(1:numLayers) = {zeros(resSize,1)};
    
    
    
    for t = 1:trainLen %for each input for training
        
        for layer = 1: numLayers % for each layer
            
            if layer == 1 % if it is the first layer, input is from the data
                input = normData(t,:)'; % first layer takes input from main input
                inputPart = Win*[input;bias];% The input part for input to first layer
            else % If layer is 2 and above
                layerStates = cellPerLayer{layer-1};  %taking the layerStates of the previous layer to serve as
                %input to the present layer
                input = layerStates(:,t); % taking the input for the layer as prevoius layer states
                inputPart = cellInterLayer{layer} *[input;bias]; % inputpart of main equation Wlayer = interlayer weights resWeights by resWeights
            end
            
            %layerStatesNew = cellPerLayer{layer};
            x = cellTrackLayerStates{layer};
            cellTrackLayerStates{layer} = (1-leaky)*x + leaky*tanh(inputPart + cellResWeights{layer}*x ); %Updating the x
            %theGuy = cellTrackLayerStates{layer};
            %layerStatesNew(:,t) = theGuy; % saving the present reserv states
            cellPerLayer{layer}(:,t) = cellTrackLayerStates{layer};            
        end
        
    end
    
    
    
    %finding the Wout First
    
    % fist generating the uniwinded reservoir activatiion conncetions to the
    % the output neuron
    unwindedReservAct = unwind(cellPerLayer, trainLen, numLayers, resSize);
    
    
    
    
    %removing washout
    unwindedReservAct = unwindedReservAct(:,washout+1:trainLen);
    yTarget = targets(washout+1:trainLen);
    
    % add input bias term
    
    unwindedReservActi = [unwindedReservAct;bias * ones(1,size(unwindedReservAct,2))];
    
    
    
    % The actual finding of Wout
    
    Wout  = yTarget' * unwindedReservActi' / (unwindedReservActi*unwindedReservActi'+reg *eye(size(unwindedReservActi,1)));
    
    
    %Testing
    %taking the last state values for each layer as starting x for the test set
    
    
    
    yPredicted = zeros(testLen,1)';
    xColumn = zeros(resSize*numLayers,1);
    
    for t = 1:testLen %for each input for training
        
        for layer = 1: numLayers % for each layer
            
            if layer == 1 % if it is the first layer, input is from the data
                input = normData(trainLen + t,:)'; % first layer takes input from main input
                inputPart = Win*[input;bias];% The input part for input to first layer
            else % If layer is 2 and above
                %input to the present layer
                input = cellTrackLayerStates{layer-1}; % taking the input for the layer as states of former layer
                inputPart = cellInterLayer{layer}*[input;bias]; % inputpart of main equation Wlayer= interlayer weights resWeights by resWeights
            end
            
            x = cellTrackLayerStates{layer};
            cellTrackLayerStates{layer} = (1-leaky)*x + leaky*tanh(inputPart + cellResWeights{layer}*x ); %Updating the x
            xColumn(1+ (layer-1)*resSize: resSize + (layer-1) * resSize) = cellTrackLayerStates{layer};
            
        end
        
        xColumnMain = [xColumn;bias];
        
        yPredicted(t) = Wout* xColumnMain ;
        
        
    end
       
    
    yReal = targets(trainLen+1:trainLen + testLen);
    mseTestTrial(trial) = sum((yReal' - yPredicted).^2)/testLen;
    timing(trial) = toc;
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
