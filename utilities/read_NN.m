function [dense, act] = read_NN(inp, NN_weights_biases, BN, act_list)

% Function to get outputs of dense, activation leyers of a neural network
% Designed for a feedforward neural network with/without batch
% 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   Input parameters:
%      inp                : input to neural network (must be samples x dim)
%      NN_weights_biases  : a structure containing weights, biases of the
%                           neural network obtained using the function
%                           get_weights_biases
%      BN                 : flag for batch normalisation
%                              0 - no BN
%                             'a'  BN after activation layer
%                             'b'  BN before activation
%      act_list           : a string containing intials of activations used 
%
%   Output parameters:
%      dense              : output of all dense layers
%      act                : activations of all layers
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

kernels = NN_weights_biases.kernels;
biases = NN_weights_biases.biases;
num_dense = length(kernels);

if BN~=0
    betas = NN_weights_biases.betas;
    gammas = NN_weights_biases.gammas;
    moving_means = NN_weights_biases.moving_means;
    moving_vars = NN_weights_biases.moving_vars;
    num_bn = length(betas);
else
    num_bn = 0;
end

feat_dim = size(inp,2);

%% Get outputs

% make features as row vectors
flag = 0;
if size(inp,1) == feat_dim
    temp_inp=inp';
else
    temp_inp=inp;    
end

bn=[]; dense=[]; act=[]; 

for pqr = 1: num_dense  
    
        dense{pqr} = bsxfun(@plus,temp_inp*kernels{pqr},biases{pqr}'); 
        if num_bn>0 & pqr<=num_bn & strcmp(BN,'b')
            bn{pqr} = batch_norm(dense{pqr}, moving_means{pqr}, moving_vars{pqr}, gammas{pqr}, betas{pqr}); % inp - samples x Num of Hidden units
            temp = bn{pqr};
        else
            temp = dense{pqr};
        end
        act{pqr} = activation(temp,act_list(pqr));
        
        if num_bn>0 & pqr<=num_bn & strcmp(BN,'a')
            bn{pqr} = batch_norm(act{pqr}, moving_means{pqr}, moving_vars{pqr}, gammas{pqr}, betas{pqr}); % inp - samples x Num of Hidden units
            temp_inp = bn{pqr};
        else
            temp_inp = act{pqr};
        end                
end
