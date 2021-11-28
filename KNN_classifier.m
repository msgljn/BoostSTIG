%%% K近邻分类器
%%%输入：TR训练集，TS测试集，K近邻
%%%输出：分类精度accuracy，分类预测的标签
function [accuracy,classify_label_TS]=KNN_classifier(TR,label_TR,TS,label_TS,K)
distance=[];
classify_label_TS=[];


[TR_row]=size(TR,1);%求TR的个数（行数）；
[TS_row]=size(TS,1);%求TS的个数（行数）；
 K_near=min(K,TR_row);%当训练集TR个数小于K近邻的K值得时候；


%计算距离
for i=1:TS_row
    for j=1:TR_row
    distance(i,j)=sqrt(sum((TR(j,:)-TS(i,:)).^2));%i表示TS中的第i个点与TR中的第j个点的距离
    end
end
%查找最小距离，判定标签
count_correct=0;

% if (TR_row>0&&TS_row>0)%%防止TR或TS是空集
for i=1:TS_row
[~,KNN_dist_position]=sort(distance(i,:));%对距离排序

for j = 1:K_near%%统计K近邻的类标签
    classify_label_KNN(j)=label_TR(KNN_dist_position(j));
end


classify_label_KNN_unique=unique(classify_label_KNN);%类标签唯一

for j = 1:length(classify_label_KNN_unique) %%统计K近邻的类标签最多的一类
   KNN_label_n(j) = sum(classify_label_KNN_unique(j) == classify_label_KNN);    
end
[~,Max_classify_label_KNN_unique_position]=max(KNN_label_n);%最多一类的位置


classify_label_TS(i)=classify_label_KNN_unique(Max_classify_label_KNN_unique_position);%预测最多的一类标签
    if classify_label_TS(i)==label_TS(i)
        count_correct=count_correct+1;
    end
end
% end
classify_label_TS=classify_label_TS';

accuracy=count_correct/TS_row;






        
