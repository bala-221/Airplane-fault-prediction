function [XValid,YValid] = validPrepare(normData,validLen,newTrainLen)

normDataValid = normData(newTrainLen+1:newTrainLen+validLen,:);



numObservations = max(normDataValid(:,1)) - min(normDataValid(:,1)) + 1 ;

YValid = cell(numObservations,1);
XValid = cell(numObservations,1);

j = min(normDataValid(:,1));

for i = 1:numObservations
    idx = normDataValid(:,1) == j;
    X = normDataValid(idx,3:end)';
    XValid{i} = X;
    Y = normDataValid(idx,2)';
    YValid{i} = Y;
    j = j + 1;
end

end
