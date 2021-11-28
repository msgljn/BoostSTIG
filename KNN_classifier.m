%%% K���ڷ�����
%%%���룺TRѵ������TS���Լ���K����
%%%��������ྫ��accuracy������Ԥ��ı�ǩ
function [accuracy,classify_label_TS]=KNN_classifier(TR,label_TR,TS,label_TS,K)
distance=[];
classify_label_TS=[];


[TR_row]=size(TR,1);%��TR�ĸ�������������
[TS_row]=size(TS,1);%��TS�ĸ�������������
 K_near=min(K,TR_row);%��ѵ����TR����С��K���ڵ�Kֵ��ʱ��


%�������
for i=1:TS_row
    for j=1:TR_row
    distance(i,j)=sqrt(sum((TR(j,:)-TS(i,:)).^2));%i��ʾTS�еĵ�i������TR�еĵ�j����ľ���
    end
end
%������С���룬�ж���ǩ
count_correct=0;

% if (TR_row>0&&TS_row>0)%%��ֹTR��TS�ǿռ�
for i=1:TS_row
[~,KNN_dist_position]=sort(distance(i,:));%�Ծ�������

for j = 1:K_near%%ͳ��K���ڵ����ǩ
    classify_label_KNN(j)=label_TR(KNN_dist_position(j));
end


classify_label_KNN_unique=unique(classify_label_KNN);%���ǩΨһ

for j = 1:length(classify_label_KNN_unique) %%ͳ��K���ڵ����ǩ����һ��
   KNN_label_n(j) = sum(classify_label_KNN_unique(j) == classify_label_KNN);    
end
[~,Max_classify_label_KNN_unique_position]=max(KNN_label_n);%���һ���λ��


classify_label_TS(i)=classify_label_KNN_unique(Max_classify_label_KNN_unique_position);%Ԥ������һ���ǩ
    if classify_label_TS(i)==label_TS(i)
        count_correct=count_correct+1;
    end
end
% end
classify_label_TS=classify_label_TS';

accuracy=count_correct/TS_row;






        
