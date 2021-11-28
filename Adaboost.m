function [a,h]=Adaboost( X,Y ,m)
%%
%% (1) initialize weight of all samples
n=size(X,1);
% m: the number of boost classifier
% n; the number of samples;
% initialize w=1/n;
Weight=zeros(m,n);
for i=1:n
    Weight(:,i)=1/n;
end
%%
a=zeros(1,m);
h=cell(m,1);
%% (2) produce training data from weight (i.e distribution)
for i=1:m
    [BX,BY,Bidx]=Bootstrap( X,Y,Weight(i,:) );
    h{i}=Bidx;
%% (3) learning a classifier from the Bootstrap, and then compute error rate on training data
    TrainingSet=BX;
    TrainingSet_Label=BY;
    TestSet=X;
    TestSet_Label=Y;
    predict = KNNC( TrainingSet,TrainingSet_Label,TestSet,3);
    error=0;
    for j=1:n
        if predict(j)~=TestSet_Label(j)
            error=error+Weight(i,j);
        end
    end
    %% (4)
    if error>0.5
        a(i)=0;  
        for j=1:n
            Weight(i+1,j)=1/n;
        end
    end
    %% (5)
    if error==0
       a(i)=10;
       for j=1:n
            Weight(i+1,j)=1/n;
       end
    end    
    %% (6)
    if error >0 && error<=0.5
        a(i)=(1/2)*log10( ( (1-error) / error) );
        for j=1:n
            if predict(j)==TestSet_Label(j)
                Weight(i+1,j)=Weight(i,j)/( 2*(1-error) );
            else
                Weight(i+1,j)=Weight(i,j)/(2*error);
            end
        end
        % normalization
        Weight(i+1,:)=Weight(i+1,:)/sum(Weight(i+1,:));
    end
end
end

