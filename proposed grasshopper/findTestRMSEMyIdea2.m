function [mseTest] = findTestRMSEMyIdea2(individual,normData,targets,outputSize,trainLen,testLen,inputSize,initLen,...
    leakyMat,resMat,inputWeights, resWeights,outputWeights,bestResEx)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


leaky = leakyMat(individual(1));
%resSize = resMat(individual(2));
x = bestResEx;
% spectralRad = spectMat(individual(3));
% resConn = resConMat(individual(4));
% reg = regMat(individual(5));
% 




%resWeights =  sprand(resSize,resSize,resConn);%-0.5;

%resWghtsMask = (resWeights~=0);

%resWeights(resWghtsMask) = (resWeights(resWghtsMask)*2-1);


% opt.disp = 0;
% rhoW = abs(eigs(resWeights ,1,'LM',opt));
% 
% value = 25;
% while  isnan(rhoW)
%     rhoW = abs(eigs(resWeights ,1,'LM','SubspaceDimension',value));
%     value = value + 3;
% end

%resWeights  = resWeights .* (spectralRad/rhoW);


%inputWeights = rand(resSize,1+inputSize)*2 -1;

%inputWeight scaling

% numCols = inputSize + 1;
% 
% for i = 1: numCols
%     presentMatrix = inputWeights(:,i);
%     esvd = svds(presentMatrix,1);    
%     index = individual(5+i);
%     presentMatrix = presentMatrix .* (inScaleMat(index)/esvd);    
%     inputWeights(:,i) =  presentMatrix;  
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% reservAct = zeros(1+inputSize+resSize,trainLen-initLen);
% 
% x = zeros(resSize,1); %initial instance of reservoir states
% 
% for t = 1:trainLen
%     u = normData(t,:)';
%     x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x );
%     if t > initLen
%         reservAct(:,t-initLen) = [1;u;x];
%     end
% end

%Wout finding
% if dropOutCon ~= 0
%     maxResActSize = size(reservAct,1);
%     numToDrop = round(dropOutCon*maxResActSize);
%     dropedRows = randperm(maxResActSize,numToDrop);
%     reservAct(dropedRows,:) = zeros;
% end

%yTarget = targets(initLen+1:trainLen);


%outputWeights = ((reservAct*reservAct' + reg*eye(1+inputSize+resSize)) \ (reservAct*yTarget))'; 

%outputWeights = yTarget'*pinv(reservAct);



%testing main
yPredicted = zeros(testLen,outputSize);

for t = 1:testLen %
    u = normData(trainLen+t,:)';
    x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x);
    ySmall =  outputWeights*[1;u;x];
    yPredicted(t) = ySmall;
end

yReal = targets(trainLen+1:trainLen+testLen);
squareErrors = (yReal-yPredicted).^2;
mseTest = sum(squareErrors)/length(yPredicted);

end

