function [mse] = findTestMsePso(outputWeights,resWeights,inputWeights,normData,targets,outSize,trainLen,testLen,leaky,bestResEx)


x = bestResEx;

%testing main
yPredicted = zeros(testLen,outSize);
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

