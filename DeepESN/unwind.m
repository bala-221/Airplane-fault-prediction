function  unwindedReservAct = unwind(cellPerLayer, trainLen, numLayers, resSize)

%UNWIND Summary of this function goes here
%   This functions picks the cell entries of every cell and generates a
%   unwinded version of all conncetions to the output nuerons(s)
unwindedReservAct = zeros(resSize*numLayers,trainLen);

for t=1: trainLen
    
    for layer=1: numLayers
        
        states =  cellPerLayer{layer};
        
        statesAtTime = states(:,t);
        
        unwindedReservAct(1+ (layer-1)*resSize: resSize + (layer-1) * resSize,t) = statesAtTime;
        
    end
    
end

end


