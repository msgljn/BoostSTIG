function   [ TrainingSet, TrainingSet_Label,TestSet,TestSet_Label]=PartitionDataSet( indices,data,t ,fold)
%%
test=find(indices==fold);
train=find(indices~=fold);
%%
TrainingSet=data(train,:);
TrainingSet_Label=t(train,:);
TestSet=data(test,:);
TestSet_Label=t(test,:);
end

