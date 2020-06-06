function [mse] = findTestMSE(inputWeights,resWeights,outputWeights,normData,targets,outputSize,trainLen,testLen,leaky,bestResEx)

%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

x = bestResEx;

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
mse = sum(squareErrors)/length(yPredicted);

end
