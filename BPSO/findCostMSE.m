function [MSE,cellWeights,x] = findCostMSE(normData,particle,reservAct,initLen,trainLen,validLen,targets,...
    outputSize,x,inputWeights,resWeights)
%FINDCOST generates the MSE of diffrent conncetions of the
%   Detailed explanation goes here
regParam = 1e-8;
leaky = 0.3;
numReservoirUnits = size(reservAct,1)-2 ;
expandedParticle = repmat(particle,[size(reservAct,2),1]);
reservoirActivationsNew = expandedParticle'.*reservAct ;
newTrainLen = round(trainLen*0.7);

%reservoirActivationsNew = reservoirActivationsNew(:,1:end-1);

yTarget = targets(initLen+1:newTrainLen);

outputWeights = yTarget'*reservoirActivationsNew' * (reservoirActivationsNew*reservoirActivationsNew'+ ...
regParam*eye(numReservoirUnits+2))^-1;



yPredicted = zeros(validLen,outputSize);

yTarget = targets(newTrainLen +1:newTrainLen +validLen);

for t = 1:validLen %
    u = normData(newTrainLen+t,:)';
    x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x);
    ySmall =  outputWeights*[1;u;x];
    yPredicted(t) = ySmall;
end


squareErrors = (yTarget-yPredicted).^2;
MSE = sum(squareErrors)/length(yPredicted);

cellWeights = cell(1,3);

cellWeights{1} = inputWeights;
cellWeights{2} = resWeights;
cellWeights{3} = outputWeights;

end

