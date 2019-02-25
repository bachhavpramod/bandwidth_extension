% Script to train a Gaussian mixture model (GMM) using 10d NB features vectors
% obtained using Conditional variational Auto-encoder (CVAE) and HB feature vectors
% 
% The 200d log-spectral coefficients from 2 neightbouring frames are
% concatenated to get 1000d vector and conditionling encoder of CVAE is
% applied to get 10d NB feature vector

% Additionally, this script gives mean square error (MSE) on test and validation data
% after estimation of HB LP coefficients using GMM regression
% 
% Written by Pramod Bachhav, August 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
% 
%   References:
% 
%     P.Bachhav, M. Todisco and N. Evans, "Latent Representation Learning for Artificial 
%     Bandwidth Extension using a Conditional Variational Auto-encoder", accepted in ICASSP 2019.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) 2019 EURECOM, France.
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

%% Get data

inp_feature = 'PS'; dimX = 200; dimY = 10;

filename = [inp_feature,'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=',num2str(dimX),'_HB=10'];
disp('Loading data')

path='./../1_Feature_extraction/';

data1 = readhtk([path,filename,'_train1']); % one frame per row
data2 = readhtk([path,filename,'_train2']); % one frame per row
data_train = [data1; data2];
data_dev = readhtk([path,filename,'_dev']); % one frame per row
data_test = readhtk([path,'TSP_',filename]); % one frame per row

clear data1 data2 data3 

% comp = 1; iter = 2;
% data_train = data_dev;
% data_test = data_dev;

l1=2; l2=2;
if strcmp(inp_feature, 'PS')
    data_train(:,1:dimX)= log(abs(data_train(:,1:dimX)));
    data_dev(:,1:dimX)= log(abs(data_dev(:,1:dimX)));
    data_test(:,1:dimX)= log(abs(data_test(:,1:dimX)));
    inp_feature = 'LPS';
end
create_data();
disp('Data loaded')

disp(['Train data :', num2str(size(data_train))])
disp(['Dev data :', num2str(size(data_dev))])
disp(['Test data :', num2str(size(data_test))])    

%% Uncomment following section to read encoder model of a VAE trained on NB LPS data

% file = 'ENC_512_256_DEC_256_512_CVAE_LPS_NB_LPC_200.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_bs=512_ep=50_5_he_n_alpha=10';
% file_temp = 'enc_X_initial_alpha=10';
% 
% path_to_CVAE_models='./../2_CVAE_training/your_models_CVAE/';
% 
% suffix1 = 'encX_mean_init';
% encX_mean = importKerasNetwork([path_to_CVAE_models,file,'_',suffix1,'.hdf5'],'OutputLayerType','regression')
% suffix2 = 'encX_var_init';
% encX_var = importKerasNetwork([path_to_CVAE_models,file,'_',suffix2,'.hdf5'],'OutputLayerType','regression')

%% Uncomment following section to read encoder model obtained after training of CVAE on NB and HB data 
% The encoder forms conditioning variable of CVAE via an auxillary network 

file = 'ENC_512_256_DEC_256_512_CVAE_LPS_NB_LPC_200.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_bs=512_ep=50_5_he_n_alpha=10';
file_temp = 'enc_X_final_alpha=10';
 
path_to_CVAE_models='./../2_CVAE_training/models_CVAE/';

suffix1 = 'encX_mean';
encX_mean = importKerasNetwork([path_to_CVAE_models,file,'_',suffix1,'.hdf5'],'OutputLayerType','regression')
suffix2 = 'encX_var';
encX_var = importKerasNetwork([path_to_CVAE_models,file,'_',suffix2,'.hdf5'],'OutputLayerType','regression')

%%
disp(['CVAE configuration ',  file])
disp(['Encoder consists of = ', suffix1, ' and ', suffix2, ' for extraction of latent variables '])

%% Sample prior from Normal distribution
rng(1); eps_x_train = randn( size(X_train_zs_mem,1), 10 );
rng(1); eps_x_dev = randn( size(X_dev_zs_mem,1), 10 );
rng(1); eps_x_test = randn( size(X_test_zs_mem,1), 10 );

%% Get latent variables (NB features) using encoder model
inp_train = reshape(X_train_zs_mem',[1,dimX*(l1+l2+1),1,size(X_train_zs_mem,1)]); % inp and inp1 are same, reshape it seems, gave diff result without transpose
zx_mu = encX_mean.predict(inp_train);
zx_log_var = encX_var.predict(inp_train);
zx_train = zx_mu + exp(zx_log_var/2) .* eps_x_train;

inp_dev = reshape(X_dev_zs_mem',[1,dimX*(l1+l2+1),1,size(X_dev_zs_mem,1)]); % inp and inp1 are same, reshape it seems, gave diff result without transpose
zx_mu = encX_mean.predict(inp_dev);
zx_log_var = encX_var.predict(inp_dev);
zx_dev = zx_mu + exp(zx_log_var/2) .* eps_x_dev;

inp_test = reshape(X_test_zs_mem',[1,dimX*(l1+l2+1),1,size(X_test_zs_mem,1)]); % inp and inp1 are same, reshape it seems, gave diff result without transpose
zx_mu = encX_mean.predict(inp_test);
zx_log_var = encX_var.predict(inp_test);
zx_test = zx_mu + exp(zx_log_var/2) .* eps_x_test;
  
%% Mean and variance normalisation

[zx_train_zs, mu_x_cvae, stdev_x_cvae] = mvn_train(zx_train);
zx_dev_zs = mvn_test(zx_dev, mu_x_cvae, stdev_x_cvae);
zx_test_zs = mvn_test(zx_test, mu_x_cvae, stdev_x_cvae);

%% GMM training using NB features with and without mean and variance normalisation

Initialization='Random';
disp(['Initialization = ',Initialization])

N = size(Y_train_zs,1); % number of frames
comp = 128; iter = 100;
disp(['Number of frames = ',num2str(N)])
disp(['Components = ',num2str(comp)])
fprintf('\n')
disp('*******************************************************')    
fprintf('\n')

%%
for j=0:1
    
if j==0
    Z=[zx_train Y_train_zs]; 
else
    Z=[zx_train_zs Y_train_zs];
end

    disp(['Building GMM ',num2str(j)]) 
    disp(['size of Z = ',num2str(size(Z))])
    fprintf('\n')
    disp(['First and last elements of first vector of Z are ',num2str(Z(1,1)),' and ',num2str(Z(end,1))])        
    tic
    OPTIONS = statset('MaxIter',iter,'Display','iter','TolFun',1e-4);
    rng(1)
    obj = gmdistribution.fit(Z,comp,'CovType','full','Regularize',1e-1,'Options',OPTIONS);  %% Z for gmdistribution.fit is FRAME X DIM
    disp('finished')
    time = toc;
    disp(['Time taken : ',num2str(time/60), ' minutes'])

fprintf('\n')
disp('*******************************************************')    

if j==1
    GMMfile_zs = [file_temp,'_zs=1','_GMM_',num2str(comp),'.mat'];
    save(GMMfile_zs, 'obj','mu','stdev','mu_x_cvae','stdev_x_cvae')

else
    GMMfile=[file_temp,'_zs=0','_GMM_',num2str(comp),'.mat'];
    save(GMMfile, 'obj','mu','stdev')
end
end

%% MSE calcuations

disp('Calculating MSE')
dim = size(Y_test, 2);

%%
for i=1:2
    
if i==1    
    disp('for zs=0')
    disp(['Loading GMM model ... = ', GMMfile])
    load(GMMfile)
    
    feat_test_cvae = zx_test;
    feat_dev_cvae = zx_dev;
    
elseif i==2
   
    disp('for zs=1')
    disp(['Loading GMM model ... = ', GMMfile_zs])
    load(GMMfile_zs)
     
    feat_test_cvae = zx_test_zs;
    feat_dev_cvae = zx_dev_zs;    
end

    ComponentMean = obj.mu';   % means for every component (every model) % (dimX+dimY) x NumOfComp
    ComponentVariance = obj.Sigma;
    apriory_prob = obj.PComponents;
    gmmr = offline_param(apriory_prob,ComponentMean,ComponentVariance,dim);

disp('MSE calculation on DEV set')
    Y_dev_est_zs=[];
    for m = 0:size(feat_dev_cvae,1)-1
        Y_dev_est_zs(m+1,:) = GMMR(feat_dev_cvae(m+1,:)', gmmr)';   
    end

    Y_dev_est = bsxfun(@times, Y_dev_est_zs, stdev_y); Y_dev_est = bsxfun(@plus, Y_dev_est, mu_y);
%     Y_dev_est = inverse_mvn(Y_dev_est_zs)  bsxfun(@times, Y_dev_est_zs, stdev_y); Y_dev_est = bsxfun(@plus, Y_dev_est, mu_y);
    fprintf('\n')
    disp(['MSE dev = ',num2str(mean(mean((Y_dev_est-Y_dev).^2)))])
    disp(['MSE gain - dev  = ',num2str(mean((Y_dev_est(:,1)-Y_dev(:,1)).^2))])
    disp(['MSE dev - on mvn data = ',num2str(mean(mean((Y_dev_est_zs-Y_dev_zs).^2)))])
    disp(['MSE gain dev - on mvn data = ',num2str(mean((Y_dev_est_zs(:,1)-Y_dev_zs(:,1)).^2))])
    fprintf('\n')   

disp('MSE calculation on TEST set')
    Y_test_est_zs =[];
    for m = 0:size(feat_test_cvae,1)-1
        Y_test_est_zs(m+1,:) = GMMR(feat_test_cvae(m+1,:)', gmmr)';   
    end

    Y_test_est = bsxfun(@times,Y_test_est_zs,stdev_y); Y_test_est = bsxfun(@plus,Y_test_est,mu_y);
    fprintf('\n')
    disp(['MSE test = ',num2str(mean(mean((Y_test_est-Y_test).^2)))])
    disp(['MSE gain - test  = ',num2str(mean((Y_test_est(:,1)-Y_test(:,1)).^2))])
    disp(['MSE test - on mvn data = ',num2str(mean(mean((Y_test_est_zs-Y_test_zs).^2)))])
    disp(['MSE gain test - on mvn data = ',num2str(mean((Y_test_est_zs(:,1)-Y_test_zs(:,1)).^2))])
    fprintf('\n')   
    disp('*******************************************************')    
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Finished')


