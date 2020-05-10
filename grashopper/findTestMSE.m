function [rmseTest] = findTestRMSE(inputWeights,resWeights,outputWeights,bestResEx,normData,targets,outputSize,trainLen,testLen,leaky,inputSize,initLen)

x = bestResEx;
% resSize = individual(1);
% spectralRadius = individual(2);
% resConn = individual(3);
% %dropOutCon = individual(4);



%resWeights =  sprand(resSize,resSize,resConn);%-0.5;

%resWghtsMask = (resWeights~=0);

%resWeights(resWghtsMask) = (resWeights(resWghtsMask)*2-1);


%opt.disp = 0;
%rhoW = abs(eigs(resWeights ,1,'LM',opt));

% value = 25;
% while  isnan(rhoW)
%     rhoW = abs(eigs(resWeights ,1,'LM','SubspaceDimension',value));
%     value = value + 3;
% end
% 
% resWeights  = resWeights .* (spectralRadius/rhoW);
% 
% 
% inputWeights = rand(resSize,1+inputSize)*2 -1;


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

%reg = 1e-8;
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
rmseTest = sum(squareErrors)/length(yPredicted);

end

