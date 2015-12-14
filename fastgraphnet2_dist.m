function [beta,beta0,conv] = fastgraphnet2(X,Y,nu,lambda1,lambda2,nhood,nn,lscaling,beta,beta0)
% -------------------------------------------------------------------------------------------
% A wrapper around a fast c-based logarithmic regression with GraphNet penalty implementation, which was inspired by the elasticlog in the 
% Donders machine learning toolbox https://github.com/distrep/DMLT. The algorithm in the mex-file is based on cyclical coordinate descent 
% akin to that in glmnet http://web.stanford.edu/~hastie/glmnet_matlab/ but taking into account the spatial constraints.  
% Should be used with a suitable "path" of regularization parameter values 
% leading to warm starts; computed automatically if nu is zero. Care has been taken to utilize the sparsity
% of the problem. The mex file implements the exact Hessian version of the
% algorithm. For investigation how much time approximation saves; see Friedman's glmnet paper. 
%  -----------------------------------------------------------------------------------------
%  (C) 2015 Jussi Tohka
%  Department of Biomedical and Aerospace Engineering 
%  Universidad Carlos III de Madrid, Spain
%  jussi dot tohka at gmail dot com
% -------------------------------------------------------------
% The method is described in 
% J. Tohka, E. Moradi, H. Huttunen. 
% Comparison of feature selection techniques in machine learning for anatomical brain MRI in dementia. 
% Neuroinformatics, in press, 2015. 
% Please cite this paper if you use the code
% Note that the paper uses different names for the parameters: lambda1 = alpha2 in the paper 
%                                                              lambda3 = alpha3 in the paper
%                                                                nu    =  lambda in the paper
%                                                                 alpha1 = 1 in the code 
% For the Graphnet penalty, please see
% Grosenick L, Klingenberg B, Katovich K, Knutson B, Taylor JE. Interpretable
% whole-brain prediction analysis with GraphNet. Neuroimage. 2013 May 15;72:304-21.
% doi: 10.1016/j.neuroimage.2012.12.062. Epub 2013 Jan 5. PubMed PMID: 23298747.
% 
% --------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software 
% for any purpose and without fee is hereby
% granted, provided that the above copyright notice appear in all
% copies.  The author and Universidad Carlos III de Madrid make no representations
% about the suitability of this software for any purpose.  It is
% provided "as is" without express or implied warranty.
% -------------------------------------------------------------  
% Note that this version is thought to be research code and it may contain strange functionalities
% At some point, a better documentation and improved code with further supported function is coming along. 
% Input parameters: 
% X: ninput x nsamples input data (required)
% Y: 1 x nsamples output data (class labels should be 0 and 1, non checking is done, required)
% nu: value-sequence for L1 penalty (set 0 if don't want provide a sequence yourself )
% lambda1: values for univariate ridge penalty 
% lambda2: values for the Laplacian ridge penalty 
% beta: initial beta
% beta0: initial offset (it is assumed that there is offset)
% nn : number of neighbours for each voxel, returned for example by getNeighbourhood.m
% nhood: neighbours of each voxel, returned, for example, by getNeighbourhood.m
% lscaling: indicates how the lambda1 and lambda2 are scaled along the
%           regularization path; [0 0] indicates that they are not scaled, 
%           [1 0] indicates that lambda1 is scaled, etc.
%           scaling means that lambda value used is lambda1*nu  
%
% beta: ninput x  length(nu) of the classifier coefficients 
% beta0: offsets ([] if not applicable)
% conv: convergence

% Parsing inputs

  standardize = 1;
  [ninput,nsamples] = size(X);
  Y = Y - min(Y);
  if standardize % standardization is necessary at the moment; don't change the value 
      Xmean = mean(X,2);
      Xstd = std(X,0,2);
      X = bsxfun(@rdivide,bsxfun(@minus,X,Xmean),Xstd);
  end    
  if nargin < 3,
    nu = 1;
  end
  if nu == 0
      nu = lambdapath(X,Y,50,0.001);
  end    
  
  if nargin < 4,
    lambda1 = 0;
  end
  
  if nargin < 5,
    lambda2 = 0;
  end
   if nargin < 8
      lscaling = [0 0];
  end
 
  if nargin < 9
    beta = zeros(ninput,1);
  end 
  
   nvar = ninput + 1;
  
  if nargin < 10,
     beta0 = 0;
  end
  
  if nargin < 7
     nhood = zeros(1,nvar);
     nn = zeros(nvar,1);
  end
 % take care that the nn(nvar) is set to 0
  nn(nvar) = 0;
  beta = [beta; beta0]; % add offset
  if(max(max(nhood)) == ninput)
      nhood = nhood - 1;
  end
 
  X = [X;ones(1,nsamples)];    % expand data with 1
   
  % calling the main function
  tic
  [beta,cnv] = elasticclog_mex332(X,double(Y),beta,lambda1,lambda2,nu,int32(nn),int32(nhood),(X.^2),lscaling > 0.5);
  toc
 
  beta0 = beta(ninput + 1,:);
  beta = beta(1:ninput,:);
  if standardize
      beta0 = bsxfun(@minus,beta0,sum(bsxfun(@times,beta,Xmean./Xstd)));
      beta = bsxfun(@rdivide,beta,Xstd);
      
  end    
  conv = cnv;
 

end

 function p = lambdapath(X,Y,nsteps,lmin)
    
      if nargin < 3, nsteps = 50; end
      if nargin < 4, lmin = 1e-4; end
      lmax = max(abs(X * (double(Y')-0.5)));
      p = exp(linspace(log(lmax),log(lmin*lmax),nsteps));
      
    end