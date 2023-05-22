%Ripley dataset
close all
clc
clear all
 load ripley.mat
% Exploring structure of data breast
figure
 histogram(Xtrain,'Normalization','probability')
figure
 hist(Xtrain)
 %This looks like a bimodal distribution
 size(Xtrain)
 % The matrix is 250 2. 2 features and 250 data points.
 % It is low dimesional
 % This could consider LDA but would need to test for the covariance and the
% distribution. If LDA solve this problem in a linear matter. An
% alternative would be to perform SVM with linear kernel and see how it
 % performs and hence learn form the dataset without looking for assumptions
 % compliance.
 % Linear
 [ gam1 , sig21 , cost1 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'lin_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'misclass'}) ;
 model=initlssvm(Xtrain,Ytrain,'classification',gam1,sig21,'lin_kernel');

 [alpha,b] = trainlssvm(model);
 [Ytest1] = simlssvm({Xtrain,Ytrain,'classsification',gam1,sig21,'lin_kernel'},{alpha,b},Xtest);
 err = sum(Ytest1~=Ytest); 
 figure
 plotlssvm({Xtrain,Ytrain,'classification',gam1,sig21,'lin_kernel'},{alpha,b});
 perf=roc(Ytest,Ytest1).*100;
 subtitle("Linear kernel")
 fprintf('\n on test: #misclass = %d, error rate = %.2f%%,error performance = %.2f%%\n', err, err/length(Ytest)*100,perf)

 % RBF
 [ gam1 , sig21 , cost1 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'misclass'}) ;
 model=initlssvm(Xtrain,Ytrain,'classification',gam1,sig21,'RBF_kernel');

 [alpha,b] = trainlssvm(model);
 [Ytest1] = simlssvm({Xtrain,Ytrain,'classsification',gam1,sig21,'RBF_kernel'},{alpha,b},Xtest);
 err = sum(Ytest1~=Ytest); 
 figure
 plotlssvm({Xtrain,Ytrain,'classification',gam1,sig21,'RBF_kernel'},{alpha,b});
 perf=roc(Ytest,Ytest1).*100;
 subtitle("RBF kernel")
 fprintf('\n on test: #misclass = %d, error rate = %.2f%%,error performance = %.2f%%\n', err, err/length(Ytest)*100,perf)

 % polynomial 
    [ gam1 , sig21 , cost1 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'poly_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'misclass'}) ;
    model=initlssvm(Xtrain,Ytrain,'classification',gam1,sig21,'poly_kernel');

 [alpha,b] = trainlssvm(model);
 [Ytest1] = simlssvm({Xtrain,Ytrain,'classsification',gam1,sig21,'poly_kernel'},{alpha,b},Xtest);
 err = sum(Ytest1~=Ytest); 
 figure
 plotlssvm({Xtrain,Ytrain,'classification',gam1,sig21,'poly_kernel'},{alpha,b});
 perf=roc(Ytest,Ytest1).*100;
 subtitle("polynomial kernel")
 fprintf('\n on test: #misclass = %d, error rate = %.2f%%,error performance = %.2f%%\n', err, err/length(Ytest)*100,perf)
% From the ripley dataset we can see that the bet model with the tuned
% hyperparameters and parameters is the LS-SVM with the RBF kernel with a
% 91.02 in performance (ROC curve) and an error rate of missclasification
% of 9%.
FolderName = 'C:\Users\toros\Desktop\homework1\ripley'  % Your destination folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  FigName   = num2str(get(FigHandle, 'Number'));
  set(0, 'CurrentFigure', FigHandle);
  saveas(gcf,fullfile(FolderName, [FigName '.png']));
end

