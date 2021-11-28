%differential evolution algorithm�㷨��DE,KNN���۹���
%%%%��ֽ����㷨����  
%%%%GS���Ż������ݼ���TR�ල�Ż��ļල��,Gm������������K���ڵ�ֵK��class_n��ĸ���,
%%%%Ҫ���������ݰ���������������У����ܲ�ͬ��������򡣵��ǵ�����ǰ�漸�к󣬾Ϳ��Բ��ð�˳��������
%%%%�������������ǩ��0,1,...)���к�(1,2,...)���ж�����,����������ǩ(-1,1)��(0,1)���п���


function [GS,label_GS,Accuracy_global]=DE_adjustment(GS,label_GS,TR,label_TR,K,Gm)  
%%
% GM��������
% K: KNN�������Ĳ���k
%%
%---------����������ݼ�ת���ɰ�����ǩ˳������-------------
[label_GS,I] = sort(label_GS);
GS_m=GS(I,:);
GS=GS_m;
[label_TR,I] = sort(label_TR);
TR_m=TR(I,:);
TR=TR_m;
%---------����������ݼ�ת���ɰ�����ǩ˳������-------------
G= 1; %��ʼ������
X=GS;
label_X=label_GS;
[X_n,D]=size(X);%��Ŀ������X�ĸ������У���ά�����У���
%%[TR_n,~]=size(TR);%��TR�ĸ�������������
%----------��ÿ�е����ֵ����Сֵ-------------
for j=1:D
    max_col_X(j)=max(X(:,j));
    min_col_X(j)=min(X(:,j));
end
%----------��ÿ�е����ֵ����Сֵ-------------
%%���й�񣬿�ѡKNN
[Accuracy_global,~]=KNN_classifier(X,label_X,TR,label_TR,K);%�Լ���д�ĵ���KNN���ۼ��㾫�ȣ�
class_n1=length(unique(label_GS));
class_n2=length(unique(label_TR));
class_n=max(class_n1,class_n2);
%%%%%%%%%%%�ֱ���X,TR�������ݼ��и��������Ŀ��ʼ
label_num_X=unique(label_X);
for jj=1:class_n1
    class_ni_X(jj)=sum(label_X==label_num_X(jj));
end
label_num_TR=unique(label_TR);
for jj=1:class_n2
	class_ni_TR(jj)=sum(label_TR==label_num_TR(jj));
end
%%%%%%%%%%%ͻ��ͽ��濪ʼ
while((G<Gm)&&(Accuracy_global<1))
    for i=1:X_n %���ÿһ��Ŀ������X����һ����������U
        %%%%�ж�X(i���ǵڼ��࿪ʼ
        temporary=0;
        class_number_X=1;
       for j=1:length(class_ni_X)
            temporary=temporary+class_ni_X(j);  
           if i>temporary
               class_number_X=class_number_X+1;
           end
       end
        %%%%�ж�X(i���ǵڼ�����
        %%%%%��ͬһ���г�ȡ����ʵ�֡�ͻ�䡱�͡����桱
              temporary=0;
              for j=1:class_number_X-1
                  temporary=temporary+class_ni_TR(j); 
              end
            rand_r=temporary+randperm(class_ni_TR(class_number_X));
            [~,rand_ri]=ismember(X(i,:),TR(rand_r,:),'rows');%�����Ƿ���rand_r(1-3)=Xi
            if rand_ri>0
            rand_r(rand_ri)=[];%%%,�����Ŵ��㷨Ҫ��rand_r(1-3)��Xi��û�п���С��3����������������
            end
            %�����ý�������[xx,a,b]=intersect(X,Y,'rows')
            %%%%%%����ͻ��ͽ��湫ʽ��ʼ          
            %%%%%%����Ӧ����ȷ��F��SFLSDE���㷨
           tau1=0.1;tau2=0.1;tau3=0.03;tau4=0.07;
           SFGSS=8;SFHC=20;Fl=0.1;Fu=0.9;
          Fi=rand(1,1);
           rand_Fi=rand(1,5);%����ӦSFLSDE�㷨��5���������
           if rand_Fi(5)<tau3
               Fi=SFGSS;
           else if tau3<=rand_Fi(5)&&rand_Fi(5)<tau4
                   Fi=SFHC;
               else if rand_Fi(2)<tau1
                        Fi=Fl+Fu*rand_Fi(1);
                    end
               end
           end
            %%%%%%����Ӧ����ȷ��F��SFLSDE���㷨��
            KK=rand(1,1);%%ͻ�������ϵ��
            if length(rand_r)>=3
            U(i,:)=X(i,:)+KK*(TR(rand_r(1),:)-X(i,:))+Fi*(TR(rand_r(2),:)-TR(rand_r(3),:));
            else                
                P_r1=X(i,:)+X(i,:).*((rand(1,D)-0.5)/5);
                P_r2=X(i,:)+X(i,:).*((rand(1,D)-0.5)/5);
                P_r3=X(i,:)+X(i,:).*((rand(1,D)-0.5)/5);
                U(i,:)=X(i,:)+KK*(P_r1-X(i,:))+Fi*(P_r2-P_r3);
            end
          %%%U��ֵ������ÿ�е����ֵ����Ϊ���е����ֵ��С����Сֵ����Ϊ��Сֵ����ʼ��
            Position_more1=find(U(i,:)>max_col_X);
            U(i,Position_more1)=max_col_X(Position_more1);
            Position_less0=find(U(i,:)<min_col_X);
            U(i,Position_less0)=min_col_X(Position_less0);
          %%%U��ֵ������ÿ�е����ֵ����Ϊ���е����ֵ��С����Сֵ����Ϊ��Сֵ���ꣻ
            label_U(i,1)=label_X(i);
           %%%%%%����ͻ��ͽ��湫ʽ��  
    end
    [Accuracy_U,~]=KNN_classifier(U,label_U,TR,label_TR,K);%�����Լ���д��KNN���ۼ��㾫��
    if (Accuracy_U>Accuracy_global)
        Accuracy_global=Accuracy_U;
        X=U;
        GS=U;
    end
    %%%%%%%ѡ���Ż���
    G=G+1;
end
        
    
    
    
    
    
    
    