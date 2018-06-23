function gmmr = offline_param(apriory_prob,ComponentMean,ComponentVariance,dimX)

% This function calculates the offline parameters needed by the function GMMR which
% perform regression using Gaussian mixture models

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   Input parameters:
%         apriory_prob        : prior probablities for mixutres
%         ComponentMean       : their corresponding means
%         ComponentVariance   : and variances
%         dimX                : dimension of input to GMMR
% 
%   Output parameters:
%         gmmr                : structure consisting of all offline parameters
%
%   References:
% 
%     K.-Y. Park and H. Kim, “Narrowband to wideband conversion
%     of speech using gmm based transformation,” in Proc. of IEEE
%     Int. Conf. on Acoustics, Speech, and Signal Processing, vol. 3,
%     2000, pp. 1843–1846.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) 2016 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gmmr.comp_prob = apriory_prob;  % apriori probabilities % can be a row or a column vector
gmmr.M_x = ComponentMean(1:dimX,:);  % (dimX) x NumOfComp 
gmmr.M_y = ComponentMean(dimX+1:end,:);   % (dimY) x NumOfComp 
gmmr.sigma_xx = ComponentVariance(1:dimX,1:dimX,:);   % CovXX  - (dimX) x (dimY) x NumOfComp 
gmmr.sigma_yy = ComponentVariance(dimX+1:end,dimX+1:end,:); % CovYY
gmmr.sigma_yx = ComponentVariance(dimX+1:end,1:dimX,:); % CovYX
gmmr.sigma_xy = ComponentVariance(1:dimX,dimX+1:end,:); % CovXY

gmmr.comp_no = length(gmmr.comp_prob);
gmmr.factor = zeros(gmmr.comp_no, 1);   
    
gmmr.invCovXX = [];
gmmr.CovXY_invCovXX = [];

for i = 1:gmmr.comp_no
    gmmr.invCovXX(:,:,i) = inv(gmmr.sigma_xx(:,:,i));
    gmmr.CovXY_invCovXX(:,:,i) = gmmr.sigma_yx(:,:,i)*gmmr.invCovXX(:,:,i);
    gmmr.posterior(i) = sqrt(det(gmmr.invCovXX(:,:,i))/((2*pi)^dimX));
end
end