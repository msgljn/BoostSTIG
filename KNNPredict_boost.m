function predict= KNNPredict_boost(a,h,TrainingSet,TrainingSet_Label,TestSet,t,K,m )
p2=zeros(m,size(TestSet,1));
predict=zeros(1,size(TestSet,1));
for j=1:m
    idx=h{j};
    p = KNNC( TrainingSet(idx,:),TrainingSet_Label(idx),TestSet,K );
    p2(j,:)=p';
end
num_class=length(unique(t));
for j=1:size(TestSet,1)
    vote=zeros(1,num_class);
    for k=1:m
        value=p2(k,j);
        vote(value)=vote(value)+a(k);
    end
    [value,predict(j)]=max(vote);
end
end

