% Script to train a Gaussian mixture model (GMM) for joint NB and HB features
% for estimation of Mutual Information (MI)
% NB vectors are log Mel filter energy (LogMFE) coefficients
% HB features are LP coefficients

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   References:
% 
%     P. Jax and P. Vary, “Feature selection for improved bandwidth
%     extension of speech signals,” in Proceedings of ICASSP, 2004.
% 
%     M. Nilsson, H. Gustaftson, S. Andersen, andW. Kleijn, “Gaussian
%     mixture model based mutual information estimation between
%     frequency bands in speech,” in Proceedings of ICASSP 2002.
% 
%     A. Nour-Eldin, T. Shabestary, and P. Kabal, “The effect of
%     memory inclusion on mutual information between speech frequency
%     bands,” in Proceedings ICASSP, 2006.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clc; close all; clear all;
addpath('./../../functions')

%% Read features
filename = 'LogMFE_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10';

dimX = 10; dimY = 10;
path='./../1_Feature_extraction/';
data = readhtk([path,filename]); % one frame per row
data = data';  % feat vect as columns

%% Apply mean-variance normalisation
stdev_x = std(data,[],2); 
mu_x = mean(data,2);
data_zs = bsxfun(@minus,data,mu_x);
data_zs = bsxfun(@rdivide,data_zs,stdev_x);

%% Train a GMM

comp = 100; iter = 128;
Initialization='Random';
disp(['Initialization = ',Initialization])
disp(['Size of original Z = ',num2str(size(data))])    

disp('*******************************************************')    
fprintf('\n')
disp(['Training GMM'])
disp(['Components = ',num2str(comp)])
disp(['Size of data = ',num2str(size(data))])
GMModel = GetGMM(data_zs',comp,Initialization,iter); % frames X dim is the input
disp('GMM training finished')

%% Calculate MI    
[fx fy fxy] = GetPDFs(data_zs(1:dimX,:),data_zs(dimX+1:end,:),GMModel);
    
disp('Calculating MI using PDF method')
I = mean(log2(fxy./(fx.*fy)));
fprintf('\n')
disp(['MI =======    ',num2str(I)])

