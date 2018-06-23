% Script to train a Gaussian mixture model (GMM) for joint NB and HB features
% NB vectors are log Mel filter energy (LogMFE) coefficients
% HB features are LP coefficients

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

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

clc; close all; clear all;

%% Read features
% filename='LogMFE_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10_train';
filename='LogMFE_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10';

dimX=10; dimY=10;
path='./../1_Feature_extraction/';
data_train = readhtk([path,filename]); % one frame per row
data_train = data_train';  % feat vect as columns

%% Apply mean-variance normalisation
temp = data_train;
stdev = std(temp')';  % examples are along rows...so need to take std along rows
mu = mean(temp')';
stdev1 = repmat(stdev,1,max(size(temp)));
mu1 = repmat(mu,1,max(size(temp)));
temp = (temp-mu1)./stdev1;
data_train = temp;

%% Train a GMM
comp = 128;
disp(['Components = ',num2str(comp)])
disp(['Numebr of frames = ',num2str(max(size(data_train)))])

OPTIONS=statset('MaxIter',100,'Display','iter','TolFun',1e-4);
rng(1)
obj = gmdistribution.fit(data_train',comp,'CovType','full','Regularize',1e-1,'Options',OPTIONS);  
disp('finished')

save(['./your_models/',filename,'_GMM_',num2str(comp),'.mat'],'obj','stdev','mu','dimX','dimY')
