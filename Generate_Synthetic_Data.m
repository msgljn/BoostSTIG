function [SyntheticData,label_SyntheticData]=Generate_Synthetic_Data(L,lable_L,U,NaNs,f,Gm)
% Gm=500;
% f=0.01;%��������
TR=[L;U];
%%
ratio=f*size(TR,1)/size(L,1);  % ����ÿһ����������Ҫ���ɵ�������Ŀ
%%
OverSampled=[];
label_OverSampled=[];
%%
for i=1:length(unique(lable_L)) %��ÿһ�����
    getFromL=find(lable_L==i);  %ȡ��ÿһ����������
    PerClass=L(getFromL,:);  % ÿһ�������
    for j=1:size(PerClass,1)
        Generated=0;
        while(Generated<ratio)
            if isempty(NaNs{getFromL(j)}) % ���ѡ���Ļ�����û����Ȼ���ڣ���ô�Ͳ����ɺϳ�����
                break;
            end
            nn=randperm(length(NaNs{getFromL(j)})); % ����һ����Ȼ���ڵ����
            Sample=PerClass(j,:); %������
            Nearest=TR(NaNs{getFromL(j)}(nn(1)),:);  %����һ����Ȼ����
            dif=Nearest-Sample;
            gap=rand(1,1);
            Synthetic=Sample+gap*dif;
            label_Synthetic=lable_L(getFromL(j));        
            OverSampled=[OverSampled;Synthetic];
            label_OverSampled=[label_OverSampled;label_Synthetic];
            Generated=Generated+1;
        end
    end
end
%%
[SyntheticData,label_SyntheticData,~]=DE_adjustment(OverSampled,label_OverSampled,L,lable_L,3,Gm);  
