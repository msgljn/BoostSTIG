function  [L,t,LER]=STDP_AdaBoost(label_x,label_x_t,unlabel_x,unlabel_x_t,test_t,Dc,m)
%% STDP with AdaBoost
%% variables
L=label_x;                 %�ѱ������
t=label_x_t;               %�ѱ�����ݵ����
U=unlabel_x;               %δ�������
C=length(unique(t));       %�������������
%% DPC
data=[L;U];
data_t=[label_x_t;unlabel_x_t];
label=[t;zeros(size(U,1),1)];
%%
arrows=DPC(data,Dc,C);
sort_idx = Find_index(arrows,L,U ); % �������������ķ������
%%
count=1;
L_index=[1:1:size(L,1)]';
U_N=0;
U_Error_N=0;
%%
while 1
    %%
    pos=find(sort_idx==count);
    if length(pos)==0
        break;
    end
    %%
    index=pos;
    classifyU=data(index,:);   
    [a,h]=Adaboost( L,t,m );
    Pre=KNNPredict_boost(a,h,L,t,classifyU,[label_x_t;unlabel_x_t;test_t],3,m);
    classifyU_t=data_t(index);
    U_N=U_N+size(classifyU,1);
    U_Error_N=U_Error_N+length(find((Pre'-classifyU_t)~=0));
    
    %% update L and U
    for i=1:length(index)
        L_index=[L_index;index(i)];
        t=[t;Pre(i)];
        label(index(i))=Pre(i);
    end
    U_index=setdiff([1:1:size(data,1)],L_index);
    L=data(L_index,:);
    U=data(U_index,:);
    %%
    count=count+1;
end
%%
LER=U_Error_N/U_N;
end