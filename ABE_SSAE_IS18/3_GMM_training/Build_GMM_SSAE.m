% Script to train a Gaussian mixture model (GMM) using joint vectors
% obtained from Semi-supervised Stacked Auto-encoder (SSAE) and HB representations
% 
% Additionally, this script gives mean square error (MSE) on test and validation data
% after estimation of HB LP coefficients using GMM regression
% 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   References:
% 
%     P.Bachhav, M. Todisco and N. Evans, "Artificial Bandwidth Extension 
%     with Memory Inclusion using Semi-supervised Stacked Auto-encoders", 
%     to appear in Proceedings of INTERSPEECH, Hyderabad, India.
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

inp_feature='LogMFE'; dimX=10; 
% inp_feature='PS'; dimX = 200;

filename = [inp_feature,'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=',num2str(dimX),'_HB=10'];
disp('Loading data')

dimX = 10; dimY = 10;
path='./../1_Feature_extraction/';
data1 = readhtk([path,filename,'_train']); % one frame per row
data2 = readhtk([path,filename,'_dev']); % one frame per row
data3 = readhtk([path,filename,'_test']); % one frame per row
data_test = readhtk([path,'TSP_',filename]); % one frame per row

data_train = [data1; data2];
data_dev = data3;

clear data1 data2 data3

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

%%
disp('************************** SSAE parameters **************************')
if strcmp(inp_feature, 'LogMFE')
    BN='b';act_list='tttttl'; NN=['6L_512_256_10_256_512_SSAE_LogMFE_NB_LPC_10.10_mem_2.2_act_tttttl_dr=0_BN=b_adam_LR=0.001_mse_ep=30_bs=512_shuff=True_he_n_alpha=0.5'];
    % BN='b';act_list='rrrrrl'; NN=['6L_512_256_10_256_512_SSAE_LogMFE_NB_LPC_10.10_mem_2.2_act_rrrrrl_dr=0_BN=b_adam_LR=0.001_mse_ep=30_bs=512_shuff=True_he_n_alpha=0.5'];
    % BN='b';act_list='tttttl'; NN=['6L_1024_512_10_512_1024_SSAE_LogMFE_NB_LPC_10.10_mem_2.2_act_tttttl_dr=0_BN=b_adam_LR=0.001_mse_ep=30_bs=512_shuff=True_he_n_alpha=0.5'];
    % BN='b';act_list='rrrrrl'; NN=['6L_1024_512_10_512_1024_SSAE_LogMFE_NB_LPC_10.10_mem_2.2_act_rrrrrl_dr=0_BN=b_adam_LR=0.001_mse_ep=30_bs=512_shuff=True__he_n_alpha=0.5'];
elseif strcmp(inp_feature, 'LPS')
    BN='b'; act_list='tttttl';
    NN='6L_512_256_10_256_512_SSAE_LPS_NB_LPC_200.10_mem_2.2_act_tttttl_dr=0_BN=b_adam_LR=0.001_mse_ep=30_bs=512_shuff=True_he_n_alpha=0.5';
    % NN='6L_1024_512_10_512_1024_SSAE_LPS_NB_LPC_200.10_mem_2.2_act_tttttl_dr=0_BN=b_adam_LR=0.001_mse_ep=30_bs=512_shuff=True_he_n_alpha=0.5';
end

path_to_NN_folder='./../2_SSAE_training/models_SSAE/';
path_to_NN_file = [path_to_NN_folder,NN,'.hdf5'];

disp(['Activations used == ' , act_list])
disp(['BN flag == ' , num2str(BN)])
disp(['SSAE file used = ',  path_to_NN_file])

%% Get output from middle layer from SSAE model
NN_weights_biases = get_weights_biases(path_to_NN_file);
[~, act] = read_NN (X_train_zs_mem, NN_weights_biases, BN, act_list);
X_train_AE = act{(3)};

%% Apply mean-variance normalisation to SSAE encoder output
mu_x_AE = mean(X_train_AE);
stdev_x_AE = std(X_train_AE);

stdev_x_AE(find(stdev_x_AE==0))= inf; 

X_train_AE_zs = bsxfun(@minus,X_train_AE,mu_x_AE);
X_train_AE_zs = bsxfun(@rdivide,X_train_AE_zs,stdev_x_AE);
      
%% Train a GMMs for SSAE NB features with and without mean-variance normalisation
disp('************************** GMM parameters **************************')
comp = 128; iter = 100;
Initialization = 'Random';
disp(['Initialization = ',Initialization])
N = size(Y_train_zs,1); 
disp(['Number of frames = ',num2str(N)])
disp(['GMM Components = ',num2str(comp)])
disp('*******************************************************')    

for j=0:1
    
    if j==0
        Z=[X_train_AE Y_train_zs]; 
    else
        Z=[X_train_AE_zs Y_train_zs];
    end

    disp(['Training GMM ',num2str(j)])
    disp(['Size of Z = ',num2str(size(Z))])
    OPTIONS = statset('MaxIter',iter,'Display','iter','TolFun',1e-4);
    rng(1)
    obj = gmdistribution.fit(Z,comp,'CovType','full','Regularize',1e-1,'Options',OPTIONS);  %% Z for gmdistribution.fit is FRAME X DIM
    disp('Training finished')
    disp('*******************************************************')    

    if j==1
        GMMfile_zs=['./GMMs_SSAE/',NN,'_zs=1','_GMM_',num2str(comp),'.mat'];
        save(GMMfile_zs,'obj','mu','stdev','mu_x_AE','stdev_x_AE')
    else
        GMMfile=['./GMMs_SSAE/',NN,'_zs=0','_GMM_',num2str(comp),'.mat'];
        save(GMMfile,'obj','mu','stdev')
    end
end

%%
disp('Calculating MSE on dev and test')
[~, act] = read_NN (X_test_zs_mem, NN_weights_biases, BN, act_list);
X_test_AE = act{3};

%% Apply mean-variance (or z-score) normalisation to SSAE test features

X_test_AE_zs=bsxfun(@minus,X_test_AE,mu_x_AE);
X_test_AE_zs=bsxfun(@rdivide,X_test_AE_zs,stdev_x_AE);
[r c] = find(isfinite(X_test_AE_zs)==0); % for relu act fn
X_test_AE_zs(r,c)=0;

[~, act] = read_NN (X_dev_zs_mem, NN_weights_biases, BN, act_list);
X_dev_AE = act{3}; 
X_dev_AE_zs = bsxfun(@minus,X_dev_AE,mu_x_AE);
X_dev_AE_zs = bsxfun(@rdivide,X_dev_AE_zs,stdev_x_AE);
[r c] = find(isfinite(X_dev_AE_zs)==0); % for relu act fn
X_dev_AE_zs(r,c)=0;

dim = 10;

%%
for i=1:2
    
if i==1    
    disp('************************** without mvn on SSAE features **************************')
    disp(['Loading GMM model ', GMMfile])
    load(GMMfile)
    
    feat_test_ae = X_test_AE;
    feat_dev_ae = X_dev_AE;
    
elseif i==2
   
    disp('************************** with mvn on SSAE features **************************')
    disp(['Loading GMM model ', GMMfile_zs])
    load(GMMfile_zs)
     
    feat_test_ae = X_test_AE_zs;
    feat_dev_ae = X_dev_AE_zs;    
end

% Get offline parameters for GMMR
    ComponentMean=obj.mu';   % means for every component (every model) % (dimX+dimY) x NumOfComp
    ComponentVariance=obj.Sigma;
    apriory_prob=obj.PComponents;
    gmmr = offline_param(apriory_prob,ComponentMean,ComponentVariance,dim);

disp('MSE calculation on DEV set')
    Y_dev_est_zs=[];
    for m=0:size(feat_dev_ae,1)-1
        Y_dev_est_zs(m+1,:) = GMMR(feat_dev_ae(m+1,:)', gmmr)';   
    end

    Y_dev_est = bsxfun(@times,Y_dev_est_zs,stdev_y); Y_dev_est=bsxfun(@plus,Y_dev_est,mu_y);
    fprintf('\n')
    disp(['MSE on validation/development data = ',num2str(mean(mean((Y_dev_est-Y_dev).^2)))])
    disp(['MSE only on gain coefficient  = ',num2str(mean((Y_dev_est(:,1)-Y_dev(:,1)).^2))])
    disp(['MSE on normalised validation/development data = ',num2str(mean(mean((Y_dev_est_zs-Y_dev_zs).^2)))])
    disp(['MSE only on gain coefficient = ',num2str(mean((Y_dev_est_zs(:,1)-Y_dev_zs(:,1)).^2))])
    fprintf('\n')   

disp('MSE calculation on TEST set')
    Y_test_est_zs=[];
    for m=0:size(feat_test_ae,1)-1
        Y_test_est_zs(m+1,:)=GMMR(feat_test_ae(m+1,:)', gmmr)';   
    end

    Y_test_est=bsxfun(@times,Y_test_est_zs,stdev_y); Y_test_est=bsxfun(@plus,Y_test_est,mu_y);
    fprintf('\n')
    disp(['MSE test data = ',num2str(mean(mean((Y_test_est-Y_test).^2)))])
    disp(['MSE only on gain coefficient  = ',num2str(mean((Y_test_est(:,1)-Y_test(:,1)).^2))])
    disp(['MSE on normalised test data = ',num2str(mean(mean((Y_test_est_zs-Y_test_zs).^2)))])
    disp(['MSE only on gain coefficient = ',num2str(mean((Y_test_est_zs(:,1)-Y_test_zs(:,1)).^2))])
    fprintf('\n')   

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Finished')


