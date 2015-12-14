% A function for computing the neighbours of a voxel given brainmask for computing the GraphNet (Laplacian L2) penalty.
% Only 6-neighbourhoods are supported at the moment
% Outputs of this function can be used as inputs to fastgraphnet2_dist.m
% OUTPUT: 
% nbr is a matrix listing the neighbours of each voxel in a brainmask specified by volmask
% nn is the number of neighbours for each voxel
% INPUT:
% volmask is the binary brain mask; should be 3-D array
% ind (optional)  is the vector of voxel indeces within the brain mask. If you give a proper brainmask, don't give this parameter 
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

function [nbr,nn] = getNeighbourhood(volmask,ind)
      
	  volmask = (volmask > 0.5);
	  if nargin < 2
	      ind = find(volmask(:));
	  end	  
	  
	  
      dims = size(volmask);
      nfeatures = length(ind);
      [x,y,z] = ind2sub(dims,ind);
      cdim = cumprod(dims);
      tdim = [1 cdim(1:(end-1))];
      
     
      nbr = zeros(nfeatures,6);
      
      for i=1:nfeatures
          
         if volmask(x(i) - 1,y(i),z(i)) > 0
           nbr(i,1) = find(ind == ((x(i) - 1) + (y(i) - 1)*dims(1) + (z(i) - 1)*dims(1)*dims(2)));
         end
         if volmask(x(i) + 1,y(i),z(i)) > 0
           nbr(i,2) = find(ind == ((x(i) + 1) + (y(i) - 1)*dims(1) + (z(i) - 1)*dims(1)*dims(2)));
         end
         if volmask(x(i),y(i) - 1,z(i)) > 0
           nbr(i,3) = find(ind == ((x(i)) + (y(i) - 2)*dims(1) + (z(i) - 1)*dims(1)*dims(2)));
         end
         if volmask(x(i),y(i) + 1,z(i)) > 0
           nbr(i,4) = find(ind == ((x(i)) + (y(i))*dims(1) + (z(i) - 1)*dims(1)*dims(2)));
         end
         if volmask(x(i),y(i),z(i) - 1) > 0
           nbr(i,5) = find(ind == ((x(i)) + (y(i) - 1)*dims(1) + (z(i) - 2)*dims(1)*dims(2)));
         end
         if volmask(x(i),y(i),z(i) + 1) > 0
           nbr(i,6) = find(ind == ((x(i)) + (y(i) - 1 )*dims(1) + (z(i))*dims(1)*dims(2)));
         end
      end
      nbr = nbr';
      nn = sum(nbr>0);
      
      for i = 1:length(nn)
           if nn(i) < 6
                tmp = nonzeros(nbr(:,i));
                nbr(:,i) = 0;
                nbr(1:nn(i),i) = tmp;
           end
      end
      nn = nn';

    
      
      
      