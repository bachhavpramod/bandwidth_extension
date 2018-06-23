% Script to train a Gaussian mixture model (GMM) for joint NB and HB features
% after memory inclusion using delta features
% NB vectors are log Mel filter energy (LogMFE) coefficients
% HB features are LP coefficients

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   References:
% 
%     A. Nour-Eldin, “Quantifying and exploiting speech memory
%     for the improvement of narrowband speech bandwidth extension,”
%     Ph.D. Thesis, McGill University, Canada, 2013.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all; 
addpath('./../../utilities')

%% Read features
filename='LogMFE_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=5_HB=5';
dimX = 5; dimY = 5;

path='../1_Feature_extraction/';
data_train = readhtk([path,filename]); % one frame per row
data_train = data_train';  % feat vect as columns

X = data_train(1:dimX,:);
Y = data_train(dimX+1:end,:);

%% Get second order delta NB and HB features 
hlen = 2;
delta_X = Deltas(X,hlen);
delta_Y = Deltas(Y,hlen);

X_new = [X; delta_X];
Y_new = [Y; delta_Y];

%% Apply mean-variance normalisation
Z=[X_new;Y_new];
fprintf('\n')

temp=Z;
stdev=std(temp')';  % examples are along rows...so need to take std along rows
mu=mean(temp')';
stdev1=repmat(stdev,1,max(size(Z)));
mu1=repmat(mu,1,max(size(Z)));
temp=(temp-mu1)./stdev1;
Z=temp;

%% Train A GMM
comp=128;
disp(['Components = ',num2str(comp)])
disp(['Size of Z = ',num2str(size(Z))])

OPTIONS=statset('MaxIter',100,'Display','iter','TolFun',1e-4);
rng(1)
obj = gmdistribution.fit(Z',comp,'CovType','full','Regularize',1e-1,'Options',OPTIONS); 
disp('finished')

save(['./your_models/',filename,'_GMM_',num2str(comp),'_mem_delta_',num2str(hlen),'.mat'],'obj','stdev','mu','dimX','dimY')
