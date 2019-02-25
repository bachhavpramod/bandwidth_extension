% Script to perform Artificial bandwidth Extension (ABE).

% Written by Pramod Bachhav, August 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

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
addpath('./../../ABE_explicit_memory_ICASSP18/3_Extension/')
addpath('./../../ABE_SSAE_IS18/4_Extension') 

%% Read speech file to be extended
path_to_file = './../Speech_files/';
file = 'F1'; % Choose any file from M1-M5 and F1-F5 

[NB Fs8] = audioread([path_to_file,'NB/',file,'_NB.wav']); 
[WB Fs16] = audioread([path_to_file,'WB/',file,'_WB.wav']);

%%
past_frames = 2; future_frames = 2; inp_feature = 'LPS';
dimX = 200; dimY = 10; reduced_dim = 10;  
path_to_models='./../2_CVAE_training/';
path_to_GMM_models = './../3_GMM_training/';

%%
dr_params.reg = 'GMMR'; comp = 128;

%%
dr_params.arch = 'PCA';   
config_PCA = 'LPS_pca_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=200_HB=10';

dr_params.zs = 0; 
GMM_file = [path_to_GMM_models, config_PCA, '_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_PCA = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

dr_params.zs = 1; 
GMM_file = [path_to_GMM_models, config_PCA, '_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_PCA_zs = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

%%
dr_params.arch = 'VAE';  
config_CVAE = 'ENC_512_256_DEC_256_512_CVAE_LPS_NB_LPC_200.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_bs=512_ep=50_5_he_n_alpha=10';

suffix = '_encX_mean_init';
dr_params.encX_mean_init = importKerasNetwork([path_to_models,'models_CVAE/',config_CVAE,suffix,'.hdf5'],'OutputLayerType','regression');
suffix = '_encX_var_init';
dr_params.encX_var_init = importKerasNetwork([path_to_models,'models_CVAE/',config_CVAE,suffix,'.hdf5'],'OutputLayerType','regression');

dr_params.zs = 0; 
GMM_file = [path_to_GMM_models, 'enc_X_initial_alpha=10_zs=', num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_VAE = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

dr_params.zs = 1; 
GMM_file = [path_to_GMM_models, 'enc_X_initial_alpha=10_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_VAE_zs = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);


%%
dr_params.arch = 'CVAE';  
config_CVAE = 'ENC_512_256_DEC_256_512_CVAE_LPS_NB_LPC_200.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_bs=512_ep=50_5_he_n_alpha=10';

suffix = '_encX_mean';
dr_params.encX_mean = importKerasNetwork([path_to_models,'models_CVAE/',config_CVAE,suffix,'.hdf5'],'OutputLayerType','regression');
suffix = '_encX_var';
dr_params.encX_var = importKerasNetwork([path_to_models,'models_CVAE/',config_CVAE,suffix,'.hdf5'],'OutputLayerType','regression');

dr_params.zs = 0; 
GMM_file = [path_to_GMM_models, 'enc_X_final_alpha=10_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_CVAE = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

dr_params.zs = 1; 
GMM_file = [path_to_GMM_models, 'enc_X_final_alpha=10_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_CVAE_zs = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

%%
dr_params.arch = 'SAE';  
config_SAE = '6L_512_256_10_256_512_SAE_LPS_NB_LPC_1000.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_mse_ep=50_pat=5_bs=512_he_n';

suffix = '_enc';
dr_params.encX = importKerasNetwork([path_to_models,'models_SAE/',config_SAE,suffix,'.hdf5'],'OutputLayerType','regression');

dr_params.zs = 0; 
GMM_file = [path_to_GMM_models, config_SAE, '_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_SAE = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

dr_params.zs = 1; 
GMM_file = [path_to_GMM_models, config_SAE, '_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_SAE_zs = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

%%
dr_params.arch = 'SSAE';  
dr_params.BN = 'b'; dr_params.act_list = 'tttttl';
config_SSAE = '6L_512_256_10_256_512_SSAE_LPS_NB_LPC_1000.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_mse_ep=50_pat=5_bs=512_he_n';

path_to_DNN_file = [path_to_models, '/models_SSAE/', config_SSAE, '.hdf5'];   

% Load the weights and biases of each layer of neural network 
dr_params.NN_weights_biases = get_weights_biases(path_to_DNN_file);

dr_params.zs = 0; 
GMM_file = [path_to_GMM_models, config_SSAE, '_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_SSAE = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

dr_params.zs = 1; 
GMM_file = [path_to_GMM_models, config_SSAE, '_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
extended_SSAE_zs = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY);

%%
l = 500;

extended_PCA = extended_PCA(l:end-l);
extended_PCA_zs = extended_PCA_zs(l:end-l);

extended_VAE = extended_VAE(l:end-l);
extended_VAE_zs = extended_VAE_zs(l:end-l);

extended_CVAE = extended_CVAE(l:end-l);
extended_CVAE_zs = extended_CVAE_zs(l:end-l);

extended_SAE = extended_SAE(l:end-l);
extended_SAE_zs = extended_SAE_zs(l:end-l);

extended_SSAE = extended_SSAE(l:end-l);
extended_SSAE_zs = extended_SSAE_zs(l:end-l);

WB = WB(l:end-l);

%% Objective mesures
% Define parameters 

Fs = 16000; 
winlen = 20*0.001*Fs; 
freq_range = 3400:10:8000; 
win = hanning(winlen); 
lpc_order = 20; 
RS = 0;

[LSD_lpc_PCA, LSD_spectrum_PCA, d_COSH_lpc_PCA, MOS_LQO_PCA] = objective_measures(WB, extended_PCA, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_PCA_zs, LSD_spectrum_PCA_zs, d_COSH_lpc_PCA_zs, MOS_LQO_PCA_zs] = objective_measures(WB, extended_PCA_zs, win, winlen/2, Fs, freq_range, lpc_order, RS);

[LSD_lpc_VAE, ~, d_COSH_lpc_VAE, MOS_LQO_VAE] = objective_measures(WB, extended_VAE, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_VAE_zs, ~, d_COSH_lpc_VAE_zs, MOS_LQO_VAE_zs] = objective_measures(WB, extended_VAE_zs, win, winlen/2, Fs, freq_range, lpc_order, RS);

[LSD_lpc_CVAE, ~, d_COSH_lpc_CVAE, MOS_LQO_CVAE] = objective_measures(WB, extended_CVAE, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_CVAE_zs, ~, d_COSH_lpc_CVAE_zs, MOS_LQO_CVAE_zs] = objective_measures(WB, extended_CVAE_zs, win, winlen/2, Fs, freq_range, lpc_order, RS);

[LSD_lpc_SAE, LSD_spectrum_SAE, d_COSH_lpc_SAE, MOS_LQO_SAE] = objective_measures(WB, extended_SAE, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_SAE_zs, ~, d_COSH_lpc_SAE_zs, MOS_LQO_SAE_zs] = objective_measures(WB, extended_SAE_zs, win, winlen/2, Fs, freq_range, lpc_order, RS);

[LSD_lpc_SSAE, LSD_spectrum_SSAE, d_COSH_lpc_SSAE, MOS_LQO_SSAE] = objective_measures(WB, extended_SSAE, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_SSAE_zs, ~, d_COSH_lpc_SSAE_zs, MOS_LQO_SSAE_zs] = objective_measures(WB, extended_SSAE_zs, win, winlen/2, Fs, freq_range, lpc_order, RS);


%%
figure;    
k = 6;
a = 200; b = 50;
bx(1) = subplot(k,2,1);
specgram(NB,[],Fs16/2)
title('Spectrogram of NB speech (Fs=8Khz)')
caxis([-a,b])
ylim([0,8000])
bx(2) = subplot(k,2,2);
specgram(WB,[],16000)
caxis([-a,b])
title('Original WB speech')
bx(3) = subplot(k,2,3);
specgram(extended_PCA,[],16000)
caxis([-a,b])
title('PCA')
bx(4) = subplot(k,2,4);
specgram(extended_PCA_zs,[],16000)
caxis([-a,b])
title('PCA + zs')
bx(5) = subplot(k,2,5);
specgram(extended_VAE,[],16000)
caxis([-a,b])
title('VAE')
bx(6) = subplot(k,2,6);
specgram(extended_VAE_zs,[],16000)
caxis([-a,b])
title('VAE + zs')
bx(7) = subplot(k,2,7);
specgram(extended_CVAE,[],16000)
caxis([-a,b])
title('CVAE')
bx(8) = subplot(k,2,8);
specgram(extended_CVAE_zs,[],16000)
caxis([-a,b])
title('CVAE + zs')
bx(9) = subplot(k,2,9);
specgram(extended_SAE,[],16000)
caxis([-a,b])
title('SAE')
bx(10) = subplot(k,2,10);
specgram(extended_SAE_zs,[],16000)
caxis([-a,b])
title('SAE + zs')
bx(11) = subplot(k,2,11);
specgram(extended_SSAE,[],16000)
caxis([-a,b])
title('SSAE')
bx(12) = subplot(k,2,12);
specgram(extended_SSAE_zs,[],16000)
caxis([-a,b])
title('SSAE + zs')

linkaxes(bx,'x')
