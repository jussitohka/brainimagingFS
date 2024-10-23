function p = lambdapath(X,Y,nsteps,lmin)
    
      if nargin < 3, nsteps = 50; end
      if nargin < 4, lmin = 1e-4; end
      lmax = max(abs(X * (double(Y')-0.5)));
%lmax = max(abs(X * (double(Y'))*10));
      p = exp(linspace(log(lmax),log(lmin*lmax),nsteps));
      
    end