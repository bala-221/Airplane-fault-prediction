function [XTest,YTest] = testPrepare(normData,testLen,trainLen)

normDataTest = normData(trainLen+1:trainLen+testLen,:);



numObservations = max(normDataTest(:,1)) - min(normDataTest(:,1)) + 1 ;

YTest = cell(numObservations,1);
XTest = cell(numObservations,1);

j = min(normDataTest(:,1));

for i = 1:numObservations
    idx = normDataTest(:,1) == j;
    X = normDataTest(idx,3:end)';
    XTest{i} = X;
    Y = normDataTest(idx,2)';
    YTest{i} = Y;
    j = j + 1;
end

end
