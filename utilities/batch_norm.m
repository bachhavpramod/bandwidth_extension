function bn = batch_norm(inp, bn_mean, bn_var, bn_gamma, bn_beta)

% Function to get output of a batch-normalisation layer
% 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   Input parameters:
%      inp                : input to BN layer (must be samples x number of hidden units)
%      bn_mean, bn_var, 
%      bn_gamma, bn_beta  : parameters of a BN layer obtained during training
%                                               
%   Output parameters:
%      bn                 : output of BN layer
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bn = bsxfun(@rdivide, bsxfun(@minus,inp,bn_mean'), sqrt(bn_var'));
bn = bsxfun(@plus, bsxfun(@times,bn,bn_gamma'), bn_beta');
