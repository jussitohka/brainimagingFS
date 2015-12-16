% Age confound removal from MRI data for machine learning
% (C) 2015 Elaheh Moradi
% Department of Signal Processing,
% Tampere University of Technology, Finland
% elaheh.moradi at tut.fi
% and Jussi Tohka
% Department of Biomedical and Aerospace Engineering 
% Universidad Carlos III de Madrid, Spain
% jussi dot tohka at gmail dot com
% --------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software 
% for any purpose and without fee is hereby
% granted, provided that the above copyright notice appear in all
% copies.  The authors, Tampere University of Technlogy, and 
% Universidad Carlos III de Madrid make no representations
% about the suitability of this software for any purpose.  It is
% provided "as is" without express or implied warranty.
% -------------------------------------------------------------  
% Xnew : MRI data (subjects x voxels matrix) age removed
% X    : Original MRI data (subjects x voxels matrix)
% age  : Age of each subject
% NCidx: Indexes of training subjects (usually normal controls)
%  IMPORTANT NOTE: Because typically class information is used while 
%                  removing the age, it is important that only the training
%                  subjects are used. For pMCI vs. sMCI classification with
%                  age removal based on normal controls,
%                  this is not an issue as the test subjects contain no 
%                  normal controls. 
% Reference:
% E. Moradi, A. Pepe, C. Gaser, H. Huttunen, and J. Tohka 
% Machine learning framework for early MRI-based Alzheimer's conversion 
% prediction in MCI subjects. NeuroImage , 104: 398 - 412, 2015.

function Xnew = ageRegression(X, age, NCidx);

X_NC = X(NCidx,:)';
age_NC= age(NCidx); % normal_data

sz = size(X_NC);
B_normal=[];
s = age_NC - sum(age_NC)/sz(2);
for i = 1:sz(1)
    respmatrix = X_NC(i,:);
    
    b(2) = sum(s.*(respmatrix' - sum(respmatrix')/sz(2)))/sum(s.^2);
    b(1) = sum(respmatrix)/sz(2) - b(2)*sum(age_NC)/sz(2);
    b2 = b';
    
    B_normal = [B_normal b2];
    
end
desmatrix = [ones(length(age),1) age ];
Y = zeros(length(age),size(B_normal,2));
Y = desmatrix*B_normal;

Xnew= X- Y;
