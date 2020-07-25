function [XTrain,YTrain] = trainPrepare(dataTrain,trainLen)

dataTrain = dataTrain(1:trainLen,:);

numObservations = max(dataTrain(:,1));

XTrain = cell(numObservations,1);
YTrain = cell(numObservations,1);
for i = 1:numObservations
    idx = dataTrain(:,1) == i;
    
    X = dataTrain(idx,3:end)';
    XTrain{i} = X;    
    Y = dataTrain(idx,2)';    
    YTrain{i} = Y;
end

end