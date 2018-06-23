function X_conc = memory_inclusion2(X,l1,l2)

% Script for EXPLICIT MEMORY INCLUSION
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   Input parameters:
%      X     :  matrix of feature vectors, feat_dim x samples
%      l1    :  number of past frames to be concatenated
%      l2    :  number of future frames to be concatenated
%
%   Output parameters:
%      X_conc  : matrix of concatenated feature vectors, feat_dim*(l1+l2+1) x samples
%
%   References:
% 
%     P.Bachhav, M. Todisco and N. Evans, "Efficient super-wide bandwidth extension
%     using linear prediction based analysis synthesis", in Proceedings of
%     ICASSP 2018, Calgary, Canada.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) 2018 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp=0;
if size(X,1)>size(X,2)
    X=X';
    temp=1;
end

dimX = min(size(X,1),size(X,2));
N = max(size(X,1),size(X,2));

X_conc=zeros((l1+l2+1)*dimX,N);

for i=1:(l1+l2+1)
    X_conc(dimX*(i-1)+1: dimX*i ,1:end-(i-1))= X(:,i:end);
end
X_conc(:,end-(l1+l2)+1:end)=[]; 

if temp==1
    X_conc = X_conc';
end
