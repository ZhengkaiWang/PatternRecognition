clear;clc;
load fisheriris.mat
X=meas;
Y=[zeros(50,1);-1*ones(50,1);ones(50,1)];


[XTrain,YTrain,XTest,YTest]=Divide(X,Y,3);

%{'setosa'}  {'versicolor'}+{'virginica'}
v1=0.01;
[Fx1,W1] = SGD(XTrain,abs(YTrain),v1);


for i = 1:size(XTrain,1)
   if YTrain(i)==-1
       YTrain(i)=0;
   end
end    

%{'setosa'}  {'versicolor'}ºÍ{'virginica'}
v2=0.001;
[Fx2,W2] = SGD(XTrain,YTrain,v2);

subplot(2,1,1);
plot(Fx1);
subplot(2,1,2)
plot(Fx2);

YReg1=p(XTest*W1);
YReg2=p(XTest*W2);
YReg=zeros(size(YReg1));

for i = 1:size(YReg1,1)
   if YReg1(i)<0.10
       YReg(i)=0;
   elseif YReg2(i)>0.90
       YReg(i)=1;
   else
       YReg(i)=-1;
   end   
end 

error=size(find(YReg-YTest~=0),1);


function [cost,W] = SGD(X,Y,v)
    W=0.00001*ones(size(X,2),1);
    K=0;%number of iteration
    cost=zeros(1000,1);
    while true
        RandSort=randperm(size(X,1));
        X=X(RandSort(1:size(X,1)),:);
        Y=Y(RandSort(1:size(Y,1)),:);
        SX=X(1:20,:);
        SY=Y(1:20,:);
        dW=((SY-p(SX*W))'*SX)';
        W=W+v*dW;
        K=K+1;
        
        cost(K)=-Y'*(X*W)+ones(1,size(X,1))*log(1+exp(X*W));
        if K>1000
            break
        end
    end

end


function rtn = p(X)
    rtn=1./(1+(exp(-X)));
end


function [XTrain,YTrain,XTest,YTest]=Divide(X,Y,n)
    Smp=size(X,1);
    RandSort=randperm(Smp);
    XTest=X(RandSort(1:Smp/n),:);
    XTrain=X(RandSort(Smp/n+1:Smp),:);
    YTest=Y(RandSort(1:Smp/n),:);
    YTrain=Y(RandSort(Smp/n+1:Smp),:);
end