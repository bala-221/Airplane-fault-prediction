function [cost,cellWeights,x] = findCostMSE(solution,normData,targets,randSelectedInputWeights,randSelectedResWeights,...
    inputWeights,resWeights,inputSize,resSize,outputSize,trainLen,initLen,leaky)


%   FindCostMSE

subInputWeights = solution(1:length(randSelectedInputWeights));
subResWeights =  solution(length(randSelectedInputWeights)+1: length(randSelectedInputWeights) + length(randSelectedResWeights));
%subFeedbackWeights = solution(length(randSelectedInputWeights)+ length(randSelectedResWeights)+1: ...
%length(randSelectedInputWeights)+ length(randSelectedResWeights)+length(randSelectedfbWeights));

inputWeights(randSelectedInputWeights) =  subInputWeights;

resWeights(randSelectedResWeights) = subResWeights;
%feedbackWeights(randSelectedfbWeights) = subFeedbackWeights ;


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


cost = mseValid;

cellWeights = cell(1,3);

cellWeights{1} = inputWeights;
cellWeights{2} = resWeights;
cellWeights{3} = outputWeights;

end
