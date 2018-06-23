% Script to train a Gaussian mixture model (GMM) for joint NB and HB features
% after memory inclusion from neighbouring frames followed by PCA
% NB vectors are log Mel filter energy (LogMFE) coefficients
% HB features are LP coefficients

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   References:
% 
%     P.Bachhav, M. Todisco and N. Evans, "Exploiting explicit memory
%     inclusion for artificial bandwidth extension", in Proceedings of
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

clc; close all; clear all;
addpath('./../../utilities')

%% Read features
filename='LogMFE_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10';

dimX = 10; dimY = 10;
path='../1_Feature_extraction/';
data_train = readhtk([path,filename]); % one frame per row
data_train = data_train';  % feat vect as columns

%% Apply mean-variance normalisation

X = data_train(1:dimX,:);

stdev_x = std(X')';  % examples are along rows...so need to take std along rows
mu_x = mean(X')';
X_zs = bsxfun(@minus,X,mu_x);
X_zs = bsxfun(@rdivide,X_zs,stdev_x);

Y = data_train(dimX+1:end,:);
stdev_y = std(Y')';  % examples are along rows...so need to take std along rows
mu_y = mean(Y')';
Y_zs = bsxfun(@minus,Y,mu_y);
Y_zs = bsxfun(@rdivide,Y_zs,stdev_y);

%% Include memory from past l1 past and l2 future frames 
% l1=1;l2=1;
l1=2;l2=2;
% l1=3;l2=3;

X_mem = memory_inclusion2(X,l1,l2);
X_zs_mem = memory_inclusion2(X_zs,l1,l2);
Y_zs = Y_zs(:,l1+1:end-l2);

mu=[mu_x;mu_y];
stdev=[stdev_x;stdev_y];   

%% Apply PCA

D=dimX;
disp('Appying PCA')
[coeff,score,latent] = pca(X_zs_mem','Centered',false,'Algorithm','eig','NumComponents',D);
X_zs_mem_pca = coeff'*X_zs_mem;
Z = [X_zs_mem_pca;Y_zs];
    
%% Train a GMM
Initialization='Random';
disp(['Initialization = ',Initialization])
disp(['Inclusion of ',num2str(l1),' past frames = and ',num2str(l1)',' future frames'])
disp(['Number of frames = ',num2str(size(Z,2))])
disp('*******************************************************')    
fprintf('\n')

comp = 128;
disp(['Components = ',num2str(comp)])
disp(['Size of Z = ',num2str(size(Z))])
OPTIONS=statset('MaxIter',100,'Display','iter','TolFun',1e-4);
rng(1)
obj = gmdistribution.fit(Z',comp,'CovType','full','Regularize',1e-1,'Options',OPTIONS);  
disp('finished')

save(['./your_models/',filename,'_GMM_',num2str(comp),'_zs_pca_',num2str(l1),'_',num2str(l2),'.mat'],'obj','mu','stdev','coeff')



