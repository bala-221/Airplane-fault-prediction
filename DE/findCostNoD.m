function [cost,cellWeights,x] = findCostNoD(individual,normData,targets,inputSize,outputSize,trainLen,initLen,leaky)

%
% FindCostMSEfor cuckoo serach old
resSize = individual(1);
spectralRadius = individual(2);
resConn = individual(3);


resWeights =  sprand(resSize,resSize,resConn);%-0.5;

resWghtsMask = (resWeights~=0);

resWeights(resWghtsMask) = (resWeights(resWghtsMask)*2-1);


opt.disp = 0;
rhoW = abs(eigs(resWeights ,1,'LM',opt));

value = 25;
while  isnan(rhoW)
    rhoW = abs(eigs(resWeights ,1,'LM','SubspaceDimension',value));
    value = value + 3;
end

resWeights  = resWeights .* (spectralRadius/rhoW);

inputWeights = rand(resSize,1+inputSize)*2 -1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

newTrainLen = round(trainLen*0.7);
validLen =  round(trainLen*0.3);

reservAct = zeros(1+inputSize+resSize,newTrainLen-initLen);

x = zeros(resSize,1); %initial instance of reservoir states

for t = 1:newTrainLen
    u = normData(t,:)';
    x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x );
    if t > initLen
        reservAct(:,t-initLen) = [1;u;x];
    end
end

%Wout finding
yTarget = targets(initLen+1:newTrainLen);
outputWeights = yTarget'*pinv(reservAct);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test for trainLen
% yPredicted = outputWeights* reservAct;
% yReal = targets(initLen+1:newTrainLen);
% squareErrors = (yReal-yPredicted').^2;
% mseTrain = sum(squareErrors)/length(yPredicted);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for valid

yPredicted = zeros(validLen,outputSize);

for t = 1:validLen %
    u = normData(t+newTrainLen,:)';
    x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x);
    ySmall =  outputWeights*[1;u;x];
    yPredicted(t) = ySmall;
end

yReal = targets(newTrainLen+1:newTrainLen + validLen);
squareErrors = (yReal-yPredicted).^2;
mseValid = sum(squareErrors)/length(yPredicted);

%cost = mseTrain + abs(mseTrain-mseValid);

cost = mseValid ;

cellWeights = cell(1,3);

cellWeights{1} = inputWeights;
cellWeights{2} = resWeights;
cellWeights{3} = outputWeights;
end