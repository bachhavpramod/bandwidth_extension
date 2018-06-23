function [fx fy fxy] = GetPDFs(X,Y,GMModel)

% Function to get Gaussian probability distribution functions fx, fy and fxy 
% using parameters defined by Gaussian mixture model 'GMModel'
% 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   Input parameters:
%      X        : input speech frame
%      Y        : frequency range over which SLP is to be performed
%      GMModel            : Sampling frequency in Hz
%
%   Output parameters:
%      fx         : PDf for row vectors in X
%      fy         : PDF for row vectors in Y
%      fxy        : A joint PDF of joint vectors Z=[X Y]
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make feature samples along rows and dimension along columns
% -------- obs1
% -------- obs2
% -------- obs3
% .
% .
% .
% -------- obsN
% mvnpdf gives prob for each row sample
% for matrix it returns Nx1 vector, with prob of each sample

[rx cx]=size(X);
if cx>rx
    X=X';
end
[ry cy]=size(Y);
if cy>ry
    Y=Y';
end

dimX=size(X,2);
Z=[X Y];

size(Z);
%%

mean_X = GMModel.mu(:,1:dimX);
mean_Y = GMModel.mu(:,dimX+1:end);

var_X = GMModel.Sigma(1:dimX,1:dimX,:);
var_Y = GMModel.Sigma(dimX+1:end,dimX+1:end,:);

fxy = zeros(size(X,1),1);
fx = zeros(size(X,1),1);
fy = zeros(size(X,1),1);

for i = 1:length(GMModel.PComponents)
    mu = GMModel.mu(i,:); 
    sigma = GMModel.Sigma(:,:,i);
    alpha = GMModel.PComponents(i);
    fxy = fxy + alpha*mvnpdf(Z,mu,sigma);
  
    fy = fy + alpha*mvnpdf(Y,mean_Y(i,:),var_Y(:,:,i));
    fx = fx + alpha*mvnpdf(X,mean_X(i,:),var_X(:,:,i));
end
