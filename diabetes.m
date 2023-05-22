% Diabetes dataset
close all
clc
clear all
%The data should be in the desktop
load diabetes.mat

% Exploring structure of data diabetes
figure
histogram(total,'Normalization','probability')
figure
hist(total)
%In overall it looks like a poisson distribution
size(total)
%The matrix is 600 8. 8 features and 600 data points.
% It is  high dimensional
%Given the dataset an SVM looks appropiate to employ because the 
 % Linear
 [ gam1 , sig21 , cost1 ] = tunelssvm ({ trainset , labels_train , 'c', [] , [] ,'lin_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'misclass'}) ;
 model=initlssvm(trainset,labels_train,'classification',gam1,sig21,'lin_kernel');
 [alpha,b] = trainlssvm(model);
 [Ytest1] = simlssvm({trainset,labels_train,'classsification',gam1,sig21,'lin_kernel'},{alpha,b},testset);
 err = sum(Ytest1~=labels_test); 
 perf=roc(labels_test,Ytest1).*100;
 subtitle("linear kernel")
 fprintf('\n on test: #misclass = %d, error rate = %.2f%%,error performance = %.2f%%\n', err, err/length(labels_test)*100,perf)

 % RBF
 [ gam1 , sig21 , cost1 ] = tunelssvm ({ trainset , labels_train , 'c', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'misclass'}) ;
 model=initlssvm(trainset,labels_train,'classification',gam1,sig21,'RBF_kernel');

 [alpha,b] = trainlssvm(model);
 [Ytest1] = simlssvm({trainset,labels_train,'classsification',gam1,sig21,'RBF_kernel'},{alpha,b},testset);
 err = sum(Ytest1~=labels_test); 
 perf=roc(labels_test,Ytest1).*100;
 subtitle("RBF kernel")
 fprintf('\n on test: #misclass = %d, error rate = %.2f%%,error performance = %.2f%%\n', err, err/length(labels_test)*100,perf)

 % polynomial 
    [ gam1 , sig21 , cost1 ] = tunelssvm ({ trainset , labels_train , 'c', [] , [] ,'poly_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'misclass'}) ;
    model=initlssvm(trainset,labels_train,'classification',gam1,sig21,'poly_kernel');

 [alpha,b] = trainlssvm(model);
 [Ytest1] = simlssvm({trainset,labels_train,'classsification',gam1,sig21,'poly_kernel'},{alpha,b},testset);
 err = sum(Ytest1~=labels_test); 
 perf=roc(labels_test,Ytest1).*100;
 subtitle("polynomial kernel")
 fprintf('\n on test: #misclass = %d, error rate = %.2f%%,error performance = %.2f%%\n', err, err/length(labels_test)*100,perf)
% From the diabetes dataset we can see that the best model with the tuned
% hyperparameters and parameters is the LS-SVM with the RBF kernel or
% linear kernel. Both yield a similar performance. Even sometimes linear
% performs better. However every run the tune hyparameters change. Yet the
% performance for RBF and linear are close to 80 % . The must unstable
% performance is the polynomial.
 FolderName = 'C:\Users\toros\Desktop\homework1\diabetes'  % Your destination folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  FigName   = num2str(get(FigHandle, 'Number'));
  set(0, 'CurrentFigure', FigHandle);
  saveas(gcf,fullfile(FolderName, [FigName '.png']));
end
