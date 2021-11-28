function  CompareOnRealdata( )
%% clear screen
clc
clear all
close all
%% load dataset
load data\SomeRealData\Wine\data
load data\SomeRealData\Wine\t
%%
ratio=0.1; % percentage of labeled sample in training set
indices = crossvalind('Kfold',size(data,1),10); % 10-fold cross-validatiob
%% 
D=sort(pdist2(data,data),'ascend');
NTr=round(length(t)*(1-ratio)); % NTr is the number of samples in the training set 
index=floor(NTr*2);
Dc=D(index);
for i=1:10
    [ labeled_x,labeled_x_t,unlabeled_x,unlabeled_x_t,TestSet,TestSet_Label,TrainingSet,TrainingSet_Label ] = Produce_SSL_DataSet( data,t,indices,i,ratio);
    fprintf('----------%gth experiment----------\n',i)
    fprintf('-----The sample number of employed real data set£º%g\n',length(t))
    fprintf('-----The sample number of L£º%g\n-----the sample number of U£º%g\n-----the sample number of test£º%g\n',length(labeled_x_t),length(unlabeled_x_t),length(TestSet_Label))
    %% STDP
    [L,L_t,LER1(i)]=STDP(labeled_x,labeled_x_t,unlabeled_x,unlabeled_x_t,Dc);
    index=KNNC(L,L_t,TestSet,3);
    Test_Accuracy1(i)=sum(TestSet_Label==index)/size(TestSet_Label,1);
    %% STDP with AdaBoost
    [L,L_t,LER2(i)]=STDP_AdaBoost(labeled_x,labeled_x_t,unlabeled_x,unlabeled_x_t,TestSet_Label,Dc,10);
    index=KNNC(L,L_t,TestSet,3);
    Test_Accuracy2(i)=sum(TestSet_Label==index)/size(TestSet_Label,1);
    %% IGNaN
    [SyntheticData,label_SyntheticData]=IGNaN(labeled_x,labeled_x_t,unlabeled_x,0.25,500);
    fprintf('-----the number of SyntheticData: %g\n',size(SyntheticData,1))
    Improved_L=[labeled_x;SyntheticData];
    Improved_L_t=[labeled_x_t;label_SyntheticData]; 
    %% STDP with AdaBoost based on IGNaN
    [L,L_t,LER3(i)]=STDP_AdaBoost(Improved_L,Improved_L_t,unlabeled_x,unlabeled_x_t,TestSet_Label,Dc,10);
    index=KNNC(L,L_t,TestSet,3);
    Test_Accuracy3(i)=sum(TestSet_Label==index)/size(TestSet_Label,1);
    
end
fprintf('----------------------------------------Experimental Result----------------------------------------\n')
fprintf('Average Accuracy of STDP:\t%g\n',mean(Test_Accuracy1)*100)
fprintf('Average Accuracy of STDP with AdaBoost:\t%g\n',mean(Test_Accuracy2)*100)
fprintf('Average Accuracy of STDP with AdaBoost and IGNaN:\t%g\n',mean(Test_Accuracy3)*100)
%%
fprintf('Average LER of STDP:\t%g\n',mean(LER1)*100)
fprintf('Average LER of STDP with AdaBoost:\t%g\n',mean(LER2)*100)
fprintf('Average LER of STDP with AdaBoost and IGNaN:\t%g\n',mean(LER3)*100)

end

