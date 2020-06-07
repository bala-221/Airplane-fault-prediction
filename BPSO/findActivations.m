function [reservAct] = findActivations(inputWeights,resWeights,normData,newTrainLen,initLen,leaky,inputSize,resSize)

reservAct = zeros(1+inputSize+resSize,newTrainLen-initLen);
x = zeros(resSize,1); %initial instance of reservoir states


for t = 1:newTrainLen
    u = normData(t,:)';
    x = (1-leaky)*x + leaky*tanh(inputWeights*[1;u] + resWeights*x );
    if t > initLen
        reservAct(:,t-initLen) = [1;u;x];
    end
end


end

