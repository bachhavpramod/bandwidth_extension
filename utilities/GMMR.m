function Y = GMMR(X, gmmr)

% This function estimates Y from X using a Gaussian mixture model trained
% using joint vectors Z = [X;Y]

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   Input parameters:
%         X            : input vector X
%         gmmmr        : structure containing parameters calculated using the function offline_params
% 
%   Output parameters:
%         Y            : estimated vector Y
%
%   References:
% 
%     K.-Y. Park and H. Kim, “Narrowband to wideband conversion
%     of speech using gmm based transformation,” in Proc. of IEEE
%     Int. Conf. on Acoustics, Speech, and Signal Processing, vol. 3,
%     2000, pp. 1843–1846.
% 
%   Acknowledgement:
%     Some part of this function was taken and modified from 
%     https://brage.bibsys.no/xmlui/handle/11250/2370992
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

weights = zeros(1,1,gmmr.comp_no);
Y = zeros(size(gmmr.M_y(:,1)));

for i = 1 : gmmr.comp_no 
    weights(i)= gmmr.comp_prob(i)*(gmmr.posterior(i)*exp(-0.5*(X-gmmr.M_x(:,i))'*gmmr.invCovXX(:,:,i)*(X-gmmr.M_x(:,i))));
end

weights = weights/sum(weights);

for i = 1: gmmr.comp_no
  Y = Y + weights(i)* (gmmr.M_y(:,i) + gmmr.CovXY_invCovXX(:,:,i)*(X - gmmr.M_x(:,i)));
end


