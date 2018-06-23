function NN_weights_biases = get_weights_biases(path_to_NN_file)

% Function to read a neural network file (.hdf5 format), saved after training 
% in keras (with thneao backend), in MATLAB
% Designed to read a feedforward neural network with/without batch
% normalisation and dropout layers
% 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   Input parameters:
%      path_to_NN_file    : path of file hdf5 file
% 
%   Output parameters:
%      NN_weights_biases  : return a structure containing all weights and biases of 
%                           layers of a feedforward neual network
%                           Also return the parameters of BN layer if
%                           exists
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

hinfo = hdf5info(path_to_NN_file);
list_layers = hinfo.GroupHierarchy.Groups(1).Groups;
list_layers_names = {list_layers.Name};

index_bn = find(contains(list_layers_names, 'batch_normalization')==1);
index_dense = find(contains(list_layers_names, 'dense')==1);
index_reg = find(contains(list_layers_names, 'reg')==1);

kernels = [];
biases = [];
for i = 1:length(index_dense)
    bias = hdf5read(path_to_NN_file,['/model_weights/dense_',num2str(i),'/dense_',num2str(i),'/bias']);
    biases{i} = bias;    
    kernel = hdf5read(path_to_NN_file,['/model_weights/dense_',num2str(i),'/dense_',num2str(i),'/kernel']);
    kernels{i} = kernel';
end

if ~isempty(index_reg)
i = i+1;
    bias = hdf5read(path_to_NN_file, '/model_weights/reg/reg/bias');
    biases{i} = bias;    
    kernel = hdf5read(path_to_NN_file, '/model_weights/reg/reg/kernel');
    kernels{i} = kernel';
    
    index_dense = [index_dense, 1];   % no need , creates error during test
end
clear kernel bias
NN_weights_biases.kernels = kernels;
NN_weights_biases.biases = biases;

if ~isempty(index_bn)

    betas = [];
    gammas = [];
    moving_means = [];
    moving_vars = [];

    for i = 1:length(index_bn)

        beta = [];
        gamma = [];
        moving_mean =[];
        moving_var = [];

        beta = hdf5read(path_to_NN_file,['/model_weights/batch_normalization_',num2str(i),'/batch_normalization_',num2str(i),'/beta']);
        gamma = hdf5read(path_to_NN_file,['/model_weights/batch_normalization_',num2str(i),'/batch_normalization_',num2str(i),'/gamma']);
        moving_mean = hdf5read(path_to_NN_file,['/model_weights/batch_normalization_',num2str(i),'/batch_normalization_',num2str(i),'/moving_mean']);
        moving_var = hdf5read(path_to_NN_file,['/model_weights/batch_normalization_',num2str(i),'/batch_normalization_',num2str(i),'/moving_variance']);

        betas{i} = beta;    
        gammas{i} = gamma;
        moving_means{i} = moving_mean;
        moving_vars{i} = moving_var;

    end
    clear beta gamma moving_mean moving_var

    NN_weights_biases.betas = betas;
    NN_weights_biases.gammas = gammas;
    NN_weights_biases.moving_means = moving_means;
    NN_weights_biases.moving_vars = moving_vars;
end
