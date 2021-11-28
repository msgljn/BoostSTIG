function  [BX,BY,index]=Bootstrap( X,Y,Weight )
%% Weight is the weight of samples and Probability of samples
n=size(X,1);
value=[1:1:n];
Prob=Weight;
index=randsrc(1,n,[value;Prob]);
BX=X(index,:);
BY=Y(index);
end

