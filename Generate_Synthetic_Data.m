function [SyntheticData,label_SyntheticData]=Generate_Synthetic_Data(L,lable_L,U,NaNs,f,Gm)
% Gm=500;
% f=0.01;%采样因子
TR=[L;U];
%%
ratio=f*size(TR,1)/size(L,1);  % 基于每一个基础样本要生成的样本数目
%%
OverSampled=[];
label_OverSampled=[];
%%
for i=1:length(unique(lable_L)) %对每一类操作
    getFromL=find(lable_L==i);  %取出每一类的样本序号
    PerClass=L(getFromL,:);  % 每一类的样本
    for j=1:size(PerClass,1)
        Generated=0;
        while(Generated<ratio)
            if isempty(NaNs{getFromL(j)}) % 如果选定的基样本没有自然近邻，那么就不生成合成样本
                break;
            end
            nn=randperm(length(NaNs{getFromL(j)})); % 其中一个自然近邻的序号
            Sample=PerClass(j,:); %基样本
            Nearest=TR(NaNs{getFromL(j)}(nn(1)),:);  %其中一个自然近邻
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
