function  feat= StabilitySelection(X,Y,M,thr,options)

% options in glmnet; 
% options.lambda: Predetermined lambda sequence
% options.alpha: Elastic-net mixing parameter
% thr: Threshold value for feature selection
% M: Number of iterations
% X: n x m data matrix, where n is the number of datapoints and m is the
% dimensionality
% Y: n x 1 datalabel.

% Stability feature selection for classification, using Glmnet package
% (C) 2015 Elaheh Moradi
% Department of Signal Processing,
% Tampere University of Technology, Finland
% elaheh.moradi at tut.fi
% -------------------------------------------------------------
% The method is described in 
% J. Tohka, E. Moradi, H. Huttunen. 
% Comparison of feature selection techniques in machine learning for anatomical brain MRI in dementia. 
% Neuroinformatics, 2015. 
% Please cite this paper if you use the code

% The method is based on stability selection algorithm described in
% J. Ye, M. Farnum, E. Yang, R. Verbeeck, V. Lobanov, N. Raghavan, ... & V. A. Narayan.
% Sparse learning and stability selection for predicting MCI to AD conversion using baseline ADNI data. 
% BMC neurology, 12(1), 46, 2012.
% --------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software 
% for any purpose and without fee is hereby
% granted, provided that the above copyright notice appear in all
% copies.  The author and Tampere University of Technology make no representations
% about the suitability of this software for any purpose.  It is
% provided "as is" without express or implied warranty.
% -------------------------------------------------------------  

sz= size(X);
n= floor(sz(1)/2);

S= cell(1,M);
for i = 1:M
    
    ind= randperm(sz(1), n);
    fit = glmnet(X(ind,:), Y(ind), 'binomial', options);
    S{i} = fit.beta;
    
end

Mat= zeros(sz(2), length(options.lambda));
for j = 1:length(options.lambda)
    
    beta= zeros(sz(2), M);
    for i = 1:M
        
        beta(:,i) =S{i}(:,j);
        
    end
    Mat(:,j)= sum(beta ~= 0,2)/M;
    
end
T=Mat';
T= max(T);
feat= find(T>thr);



