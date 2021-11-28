function [SyntheticData,label_SyntheticData]=IGNaN(L,lable_L,U,f,Gm)
%% instance generation with natural neighbors (IGNaN)
%% NaN_Search
NaNs=NaN_Search([L;U]);
%% Generation with NaN, Then DE
[SyntheticData,label_SyntheticData]=Generate_Synthetic_Data(L,lable_L,U,NaNs,f,Gm);
%%
end

