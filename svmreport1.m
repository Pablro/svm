close all
clc
clear all

% Report 1
% Training
 X1=randn (50 ,2)+1;
 X2 = randn (51 ,2)-1;
 % test
  X3=randn (50 ,2)+1;
 X4 = randn (51 ,2)-1;
 % Class labels
 Y1 = ones (50 ,1);
 Y2= -ones (51 ,1);
 % The two Gaussian distributions
 figure;
 hold on
 histogram(X1,9,'Normalization','probability');
 histogram(X2,9,'Normalization','probability');
 
 hold off
% My analysis is the following: as both distributions are extracted from the standard normal distribution
% of the MATLAB  function this should have a mean of 0 and variance of 1.
% However this is not totally true because the shift cause on the means because the addition or substraction of 1.
% How complex the formulation should be
% This will comply with condtions for applying linear discriminant analysis because the shift of means.
% Yet there is still the question on how to proof that both preserve the same covariance matrix in order to fullfill with the prerequisites
% for applying discriminant analysis. 
% Yet we know both distributions were extracted from the same distribution and represent a gaussian mixture distribution and the shift of means is observed but
% assessing the equality of the covariance distribution I found it tricky.
% The right way to proceed is to test for both covariances matrices.
% I will implement a linear discriminant classifier
 cov(X1)
 cov(X2)
 %This covariances should be tested to proof that are equal. An assumption is taken because the sampling method. 
 meanX1=mean(X1)
 meanX2=mean(X2);
 X=[X1; X2];
 % the variances is 1 because the sampling was taken from N(0,1)
 s2=1;
 % Prior distribution for this case is the class relative frequencies
 P1=(length(X1)/length(X))
 P2=(length(X2)/length(X))
 % A constant of the logarithms from the discriminant equation 
 logAB=log(P1/P2);
 % the weights of the linear discriminant function
 w=meanX2-meanX1;
 % The intercept of LDA
 b=-1/2.*(meanX2+meanX1).*(w)+logAB
 % LDA function equvalent to dA(x)-dB(x)=0
 f = @(x1,x2) b(:,1)+b(:,2) + w(:,1)*x1 + w(:,2)*x2;
 figure;
 hold on;
 plot(X3(:,1),X3(:,2),'ro')
 plot(X4(:,1),X4(:,2),'bo')
 % LDA boundary plot
 fimplicit(f,'g')
 hold off
% Exercise 1.2
% a) the hyperplane for the SVM with linear kernel is shifting depending on
% the datapoints loccation. The SVM it seems to be balancing for the
% missclasified points with the slacks variables. At the same time the
% margin can also be affected depending on the closest support datapoints
% closest to the equation. But in overall the number of datapoints affect
% how to locate the hyperplane to classify the data, highlighting how SVM
% rely completely on the support vectors location for the classification
% analysis.

% b) the C parameter control the trade of of between
% maximizing the margin and minimissing the miss classification hinge loss
% function by controlling the slack variable. (simple to observe with a linear kernel)
% In the other hand RBF kernel. 
% c)If sigma is small, closest to 0 the decision boundary is not very clear
% as it will become near to indetermined. This is observed because when
% sigma is closest to 0 no clear decision boundary is seen around the
% points and just a color background is observed. How ever when the sigma
% is small but large enough to be significantly different from 0 then a non
% linear decision boundary is observed. Otherwise when the sigma is too
% large a linear decision boundary is approach. Another important
% observation is that when sigma increase similar label and close support points will
% tend cluster under a same decision boundary while when sigma decrease
% each point will tend to be drawn under the same decision boundary. The C
% weight  is still performing a similar function as the linear svm kernel
% as it is putting enphasis for hardening or softening for missclassified
% support points and considering in the decsion boundary.
% It is clear that SVM with RBF kernel perfoms much better as approach to
% classify missclassified non separable data with a linear kernel with a
% non linear kernel as the RBF by which choosing the right tuning
% parameters makes the data being correclty classified. However the
% complexity of the non linear by choosing the rigth parameters values is what might be discuss.
% Questions
% A support vector at least taking visual example of this exercise, and
% using a linear kernel (just for simplfying the exmplantion) it is a data
% point close to the hyperplane (within the margin) that helps to draw the decision boundary
% for classifying the complete data. Noting that less support vectors are
% considered when giving a higher weight(smaller margin) but considering missclassified data points  or softing
% the margins for missclassification. Hence considering the importance of a
% support vector is explained by softening or stregthening the margin.
% Why?


% 1.3
 % Poly kernel
 % loading data
  load iris.mat
   % different polynomial orders.
  for degree=1:10
      gam= 1;
     type='classification'
     [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[1,degree],'poly_kernel'});
     Ytest1 = simlssvm({Xtrain,Ytrain,type,gam,[1,degree],'poly_kernel'},{alpha,b},Xtest);
     figure
     plotlssvm({Xtrain,Ytrain,type,gam,[1,degree],'poly_kernel'},{alpha,b});
     err = sum(Ytest1~=Ytest); 
    fprintf('\n on test: #misclass = %d, error rate = %.2f%%, pol= %d\n', err, err/length(Ytest)*100,degree)
     disp('Press any key to continue...'), pause,   
  end
% Overfitting observation is what has to be observed when increasing the
% degree of the polinomial.
 

% RBF kernel
% Fixing gam
 errlist=[];
 gam= 1;
 sig2=[0:0.02:1]
 type='classification'
 for index=sig2
       [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,index,'RBF_kernel'});
       Ytest1 = simlssvm({Xtrain,Ytrain,type,gam,index,'RBF_kernel'},{alpha,b},Xtest);
       figure
      plotlssvm({Xtrain,Ytrain,type,gam,index,'RBF_kernel'},{alpha,b});
       err = sum(Ytest1~=Ytest); 
       errlist=[errlist err];
      fprintf('\n on test: #misclass = %d, error rate = %.2f%%, sigma=%.2f%%\n', err, err/length(Ytest)*100,index)
       disp('Press any key to continue...'), pause,  
 
      % It seems to be 0.04  
 end
 figure;
 plot(log(sig2), errlist, '*-'), 
 xlabel('log(sig2)'), ylabel('number of misclass'),
% % Same now but for gam
 gam= [0.1:0.1:1];
 sig2=0.04
 type='classification'
 errlist=[];
 for index=gam
     [alpha,b] = trainlssvm({Xtrain,Ytrain,type,index,sig2,'RBF_kernel'});
      Ytest1 = simlssvm({Xtrain,Ytrain,type,index,sig2,'RBF_kernel'},{alpha,b},Xtest);
      figure
      plotlssvm({Xtrain,Ytrain,type,index,sig2,'RBF_kernel'},{alpha,b});
      err = sum(Ytest1~=Ytest);
      errlist=[errlist err];
     fprintf('\n on test: #misclass = %d, error rate = %.2f%%, gam=%.2f%%\n', err, err/length(Ytest)*100,index);
      disp('Press any key to continue...'), pause,   
 end
 figure;
 plot(gam, errlist, '*-'), 
 xlabel('gam'), ylabel('number of misclass'),
% In general it seems that C or gamma have a error rate of 0 above a value
% of 0.9

% 1.3.2 Tuning parameters using validation:
% Random split
gam=0.9;
sig2=0.04;
% This represents the generalization performance.
  perf = rsplitvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , 0.80 , 'misclass') 
% k fold crossvalidation
% This represents the generalization performance.
 perf = crossvalidate ({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'} , 10 , 'misclass') 
% Leave one out validation
perf= leaveoneout ({ Xtrain , Ytrain ,'c',gam,sig2,'RBF_kernel'} ,'misclass') 
performances=[]
% value selection
sigma2=[1:1:10];
gam=[1:1:10];
for indexsig=sigma2
    for indexgam=gam
        perf1 = rsplitvalidate ({ Xtrain , Ytrain , 'c', indexgam , indexsig ,'RBF_kernel'} , 0.60 , 'misclass') ;
        perf2 = crossvalidate ({ Xtrain , Ytrain , 'c', indexgam , indexsig ,'RBF_kernel'} , 10 , 'misclass') ;
        perf3= leaveoneout ({ Xtrain , Ytrain ,'c',indexgam,indexsig,'RBF_kernel'} ,'misclass') ;
        trial=[indexgam,indexsig,perf1,perf2,perf3];
        if isempty(performances)
            performances=[performances  trial];


        else
            performances=[performances ; trial];

        end
        
    end

end
% Best crossvalidate performance parapeters
[val,idx]=bestperf(performances(:,4));
order=[performances(idx,1), performances(idx,2),val];
fprintf("\ncrossvalidate values")
fprintf('\ngam: %1$.2f',order(1))
fprintf('\nsigma: %1$.2f',order(2))
fprintf('\nperformance: %1$.2f',order(3))
% Best leave one out performance parapeters
[val,idx]=bestperf(performances(:,5));
order=[performances(idx,1), performances(idx,2),val];
fprintf('\nleave one out values')
fprintf('\ngam: %1$.2f',order(1))
fprintf('\nsigma: %1$.2f',order(2))
fprintf('\nperformance: %1$.2f',order(3))
% Best randomsplit performance parapeters
[val,idx]=bestperf(performances(:,3));
order=[performances(idx,1), performances(idx,2),val];
fprintf('\nrandomsplitvalues')
fprintf('\ngam: %1$.2f',order(1))
fprintf('\nsigma: %1$.2f',order(2))
fprintf('\nperformance: %1$.2f\n',order(3))
% In overall and in comparison with the other validation techniques the
% performance of randomsplit yields the highest performance. Yet 
% crossvalidation is preferred becuase it usually train-validate over several splits of the
% data (in relation with the k groups) allowing to know more on how well
% model could fit on unseen data. the random split only always work on two
% groups the training part and the validating part which also might be
% affected by the proportion of splitting hence the size of the data.
% Because of this is logical to think that randomsplit will yield a higher
% performance as it is evaluating on less split than leave one out and
% crossvalidation method. Hence crossvalidation or leave one out  will be a more interesting
% result to consider for validating our models on unseen data.
% https://medium.com/@eijaz/holdout-vs-cross-validation-in-machine-learning-7637112d3f8f
% https://towardsdatascience.com/understanding-8-types-of-cross-validation-80c935a4976d
% The K seems to be standarly between 10 or 5 but it is usually associated
% with higher computational cost and a trade off between bias and variance
% of the true error given different new dataset. Yet this seems to be a
% more emprical approach given the formulate problem. but stardarly 10 or 5
% is usally choose.
% Automating parameter tuning

 [ gam1 , sig21 , cost1 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'simplex', 'crossvalidatelssvm',{10 , 'misclass'}) ;
 [ gam2 , sig22, cost2 ] = tunelssvm ({ Xtrain , Ytrain , 'c', [] , [] ,'RBF_kernel'} , 'gridsearch', 'crossvalidatelssvm',{10 , 'misclass'}) ;
 fprintf('\nsimplex')
 fprintf('\ngam: %1$.2f',gam1)
 fprintf('\nsigma: %1$.2f',sig21)
 fprintf('\nperformance: %1$.2f\n',cost1)
 fprintf('\ngridsearch')
 fprintf('\ngam: %1$.2f',gam2)
 fprintf('\nsigma: %1$.2f',sig22)
 fprintf('\nperformance: %1$.2f\n',cost2)
% The difference between both is that simplex tuning yields a slighlty lower
% cost and their values deviates in their parameters.
% The computational speed difference is not appreciable as the process end
% fast. Yet for the optimization of the parameters it is more efficient than doing by hand in
% the sense of different values explorations. In other words programming by
% hand the parameters by means of iterations (for loops) and then searching
% for the highest score it is simply inefficient and it is well known to be
% associated in the increasing computational complexity. Solving this in matlab for purpose of excercise means reducing the search space
% for accelerating calculations or the computational cost. In that sense
% this automated tunning method higlights the similar costs and parameters by coincidence
% but with much better computational complexity (less time) and considering
%  the general optimal parameters of the given problem and not just a possible local
%  optimum as it was search by hand.
% The differnece in the iterations is explain by the method itself. And it
% is related on how the parameters are optimized. First a state of the art
% technique is used for determining the paremeters according to certain criterrion solving a  global
% optimzation problem , such teecqhnique is called Coupled Simulated Annealing
% (CSA).This will be the first run. The second run is then the refinement
% of such parameter by another optimization via a simpelx or gridsearch
% procedure. Resulting in the differences about the second run.
[ alpha , b ] = trainlssvm ({ Xtrain , Ytrain , 'c', gam1, sig21 ,'RBF_kernel'}) ;
Ytest1 = simlssvm ({ Xtrain , Ytrain , 'c', gam1 , sig21 ,'RBF_kernel'} , { alpha , b } , Xtest ) ;
roc ( Ytest ,Ytest1 ) 

% The performance of the test is 0.9 which is quite good.
% We usually do not do it on the practice set because we usually like to observe
% how well our model perform on unseen data and not in the data that was
% train. This is is because generalization. Otherwise would not be
% meaningful.

 bay_modoutClass ({ Xtrain , Ytrain , 'c', gam1 , sig21 } , 'figure') ;
colorbar
bay_modoutClass ({ Xtrain , Ytrain , 'c', 5 , 25 } , 'figure') ;
colorbar
% The surface plot showing colors gradient in the plot represent the
% different values of the posterior class probability of the classifier.
% The values of sigma and gam affect how such gradient or surface of colors
% is display. In overall and regarding the classifier, while more refine
% are the hyperparameters gm and sig2 the surface display a more
% distigishable color graident with respect to the label of the data. This
% means that a color representing a higher density or a posterior class
% probability close to 1 represent one label and a color with a density
% closer to 0 next to the other label. Hence when gam and sigm2 are more
% refine the colors are better delimited according to the actual label.
% Otherwise when the parameters are not that refine the surface become more
% homogeneus meaning the difficutlties of the classifier from distiguish
% the data to their respective labels. Hence highlighting a possible
% increase in misclassifed data as the posterior class porbabilityb is
% decrease or increase.
FolderName = 'C:\Users\toros\Desktop\homework1\exercises'  % Your destination folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  FigName   = num2str(get(FigHandle, 'Number'));
  set(0, 'CurrentFigure', FigHandle);
  saveas(gcf,fullfile(FolderName, [FigName '.png']));
end
function [val,idx]= bestperf(a)
[val, idx] = min(a);
end
