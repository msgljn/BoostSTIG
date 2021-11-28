function  [NaN,NaNE]=NaN_Search(data)
%% NaN_search and Seach_LS 
%% initialize paramters
n=size(data,1);
r=1;
tag=1;
%%
NaN=cell(n,1)';
RN=zeros(n,1);
KNN=cell(n,1)';
RNN=cell(n,1)';
%%
kdtree=KDTreeSearcher(data,'bucketsize',1); % 
index = knnsearch(kdtree,data,'k',n);%
index(:,1)=[];
%%
while tag
    % rth round for search KNN and RNN
    KNN_idx=index(:,r);
    %% Compute RNN and KNN
    for i=1:n
        RNN{KNN_idx(i)}=[RNN{KNN_idx(i)};i];
        KNN{i}=[KNN{i};KNN_idx(i)];
    end
    %% 
    pos=[];
    for i=1:n
        if ~isempty(RNN{i})  
           pos=[pos;i];
        end
    end
    RN(pos)=1;
    %% stopping condition
    cnt(r)=length(find(RN==0));
    if r>2 && cnt(r)==cnt(r-1)
       tag=0;
       r=r-1;
    end
    r=r+1;
end
for i=1:n
    NaN{i}=intersect(RNN{i},KNN{i});
end
NaNE=r;
end