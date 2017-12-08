clear;clc;

%Load data set
load fisheriris.mat
X=meas;
Y=[ones(50,1);2*ones(50,1);3*ones(50,1)];

%Divide data set to Train-Test
[XTrainA,YTrainA,XTestA,YTestA]=Divide(X(1:50,:),Y(1:50,:),3);
[XTrainB,YTrainB,XTestB,YTestB]=Divide(X(51:100,:),Y(51:100,:),3);
[XTrainC,YTrainC,XTestC,YTestC]=Divide(X(101:150,:),Y(101:150,:),3);

%Set the parameters of SGD
vAB=0.1;
vBC=0.1;
vAC=0.1;
KmaxAB=100;
KmaxBC=100;
KmaxAC=100;

tic;
%Train the classifier of AB AC and BC
[CostAB,WAB] = SGD([XTrainA;XTrainB],[zeros(size(XTrainA,1),1);ones(size(XTrainB,1),1)],vAB,KmaxAB);
[CostBC,WBC] = SGD([XTrainB;XTrainC],[zeros(size(XTrainB,1),1);ones(size(XTrainC,1),1)],vBC,KmaxBC);
[CostAC,WAC] = SGD([XTrainA;XTrainC],[zeros(size(XTrainA,1),1);ones(size(XTrainC,1),1)],vAC,KmaxAC);
toc;

subplot(3,1,1);
plot(CostAB);
xlabel('AB Classifier');
ylabel('Cost fuction');
subplot(3,1,2);
plot(CostBC);
xlabel('BC Classifier');
ylabel('Cost fuction');
subplot(3,1,3);
plot(CostAC);
xlabel('AC Classifier');
ylabel('Cost fuction');

%Test classifier
XTest=[XTestA;XTestB;XTestC];
YTest=[YTestA;YTestB;YTestC];
YRegAB=p(XTest*WAB);
YRegBC=p(XTest*WBC);
YRegAC=p(XTest*WAC);
YReg=vote(YRegAB,YRegBC,YRegAC);
Error=size(find(YReg-YTest~=0),1);
Acc=1-(Error/size(YTest,1));


function [Cost,W] = SGD(X,Y,v,Kmax)
    W=-10*ones(size(X,2),1);
    K=0;%number of iteration
    KFlag=1;
    Cost=zeros(Kmax,1);
    while true  
        RandSort=randperm(size(X,1));
        X=X(RandSort(1:size(X,1)),:);
        Y=Y(RandSort(1:size(Y,1)),:);
        SX=X(1:31,:);
        SY=Y(1:31,:);
        dW=-((SY-p(SX*W))'*SX)';
        W=W-v*dW;
        K=K+1;
        Cost(K)=-Y'*(X*W)+ones(1,size(X,1))*log(1+exp(X*W));
        if K>10*KFlag
            v=0.1*v;
            KFlag=K;
        end
        if K>Kmax
            break
        end
    end
end


function YReg = vote(YRegAB,YRegBC,YRegAC)
    YRegTmp=zeros(size(YRegAB,1),3);
    for i = 1:size(YRegAB,1)
        if YRegAB(i)<0.05
            YRegTmp(i,1)=YRegTmp(i,1)+1;
        end
        if YRegAB(i)>0.95
            YRegTmp(i,2)=YRegTmp(i,2)+1;
        end   
    end 
    for i = 1:size(YRegBC,1)
        if YRegBC(i)<0.05
            YRegTmp(i,2)=YRegTmp(i,2)+1;
        end
        if YRegBC(i)>0.95
            YRegTmp(i,3)=YRegTmp(i,3)+1;
        end   
    end 
    for i = 1:size(YRegAC,1)
        if YRegAC(i)<0.05
            YRegTmp(i,1)=YRegTmp(i,1)+1;
        end
        if YRegAC(i)>0.95
            YRegTmp(i,3)=YRegTmp(i,3)+1;
        end   
    end 

    YReg=zeros(size(YRegAB));
    for i = 1:size(YRegTmp,1)
        if YRegTmp(i,1)==2
            YReg(i)=1;
        end
        if YRegTmp(i,2)==2
            YReg(i)=2;
        end 
        if YRegTmp(i,3)==2
            YReg(i)=3;
        end
    end

end

function rtn = p(X)
    rtn=1./(1+(exp(-X)));
end


function [XTrain,YTrain,XTest,YTest]=Divide(X,Y,n)
    Smp=size(X,1);
    RandSort=randperm(Smp);
    XTest=X(RandSort(1:ceil(Smp/n)),:);
    XTrain=X(RandSort(ceil(Smp/n)+1:Smp),:);
    YTest=Y(RandSort(1:ceil(Smp/n)),:);
    YTrain=Y(RandSort(ceil(Smp/n)+1:Smp),:);
end