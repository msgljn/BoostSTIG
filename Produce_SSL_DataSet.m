function [ labeled_x,labeled_x_t,unlabeled_x,unlabeled_x_t,TestSet,TestSet_Label,TrainingSet,TrainingSet_Label ] = Produce_SSL_DataSet( data,t,indices,fold,ratio)
%%
[ TrainingSet, TrainingSet_Label,TestSet,TestSet_Label]=PartitionDataSet( indices,data,t ,fold);
%%
labeled_x=[];
labeled_x_t=[];
unlabeled_x=[];
unlabeled_x_t=[];
%%
for i=1:length(unique(t))
    pos=find(TrainingSet_Label==i);
    n=ceil(length(pos)*ratio);
    labeled_x=[labeled_x; TrainingSet(pos(1:n),:)];
    labeled_x_t=[labeled_x_t;TrainingSet_Label(pos(1:n))];
    unlabeled_x=[unlabeled_x;TrainingSet(pos(n+1:end),:)];
    unlabeled_x_t=[unlabeled_x_t;TrainingSet_Label(pos(n+1:end))];
end
end

