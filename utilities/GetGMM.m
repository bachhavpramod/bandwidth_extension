function GMModel = GetGMM(Z,comp,Initialization,iter)
% Function for to train a Gaussian mixture model (GMM)
% 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   Input parameters:
%      Z               : matrix of vectors (dim X samples)
%      comp            : number of GMM components
%      Initialization  : Intialisation method
%                            'Random' 
%                            'kmeans'
%                            'kmeansLBG'
%      iter            : number of iterations
% 
%   Output parameters:
%      GMModel         : object comtaining GMM parameters
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

if strcmp(Initialization,'kmeans')
    rng(1) ; 
    [~,C] = kmeans(Z,comp);  
    disp('GMM initialised using kmeans')
elseif strcmp(Initialization,'kmeansLBG')
    rng(1) ; 
    [C] = kmeanlbg(Z,comp);
    disp('GMM initialised using kmeansLBG')
end

OPTIONS=statset('MaxIter',iter,'Display','iter','TolFun',1e-4);

if strcmp(Initialization,'Random')
    rng(1) ; % added on 28 July to have fixed random initialisation
    disp('Random Initialisation used for GMM')
    GMModel = gmdistribution.fit(Z,comp,'CovType','full','Regularize',1e-1,'Options',OPTIONS);
else
    covZ=cov(Z);
    for i=1:comp
        initialSigma(:,:,i) = covZ;
    end
    initialWeights = ones(1,comp)/comp;
    S.mu = C;
    S.Sigma = initialSigma;
    S.PComponents = initialWeights;
    GMModel=gmdistribution.fit(Z',comp,'CovType','full','Start',S,'Regularize',1e-1,'Options',OPTIONS);  
end

