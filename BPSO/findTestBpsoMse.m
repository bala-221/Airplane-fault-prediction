function [MSE] = findTestBpsoMse(normData,outputWeights,trainLen,testLen,targets,...
    outputSize,inputWeights,resWeights,bestResEx)
%FINDCOST generates the MSE of diffrent conncetions of the
%   Detailed explanation goes here
%regParam = 1e-8;
leaky = 0.3;
x = bestResEx;

yPredicted = zeros(testLen,outputSize);


yTarget = targets(trainLen+1:trainLen+testLen);


for t = 1:testLen %
    u = normData(trainLen+t,:)';
    x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x);
    ySmall =  outputWeights*[1;u;x];
    yPredicted(t) = ySmall;
end

squareErrors = (yTarget-yPredicted).^2;
MSE = sum(squareErrors)/length(yPredicted);

end

