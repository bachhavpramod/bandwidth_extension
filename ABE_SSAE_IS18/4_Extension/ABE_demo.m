% Script to perform Artificial bandwidth Extension (ABE).

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

%% Read speech file to be extended
path_to_file = './../Speech_files/';
file = 'M5'; % Choose any file from M1-M5 and F1-F5 

[NB Fs8] = audioread([path_to_file,'NB/',file,'_NB.wav']); 
[WB Fs16] = audioread([path_to_file,'WB/',file,'_WB.wav']);

past_frames=2; future_frames=2; 

addpath('./../../ABE_explicit_memory_ICASSP18/3_Extension/')

%% To perform ABE using the baseline  
% (for further details refer - P.Bachhav, M. Todisco and N. Evans, "Exploiting explicit memory inclusion for 
% artificial bandwidth extension", in Proceedings of ICASSP 2018, Calgary, Canada
inp_feature = 'LogMFE'; dimX = 10; dimY = 10; 
dr_params.arch = 'PCA';
extended_speech_M2 = ssae_dr_abe(NB, inp_feature, past_frames, dr_params, future_frames, dimX, dimY);

%% To perform ABE using dimensionality reduction with SSAE

% with LogMFE as input features (Arch-2C with 'tanh' activations)
inp_feature = 'LogMFE'; dimX = 10; dimY = 10;
dr_params.arch = '6L_1024_512_10_512_1024_SSAE'; dr_params.BN = 'b'; dr_params.act_list = 'tttttl'; dr_params.zs = 1;
ext_Arch_2C_logmfe = ssae_dr_abe(NB, inp_feature, past_frames, dr_params, future_frames, dimX, dimY);

% with LPS as input features (Arch-2C with 'tanh' activations)
inp_feature = 'LPS'; dimX = 200; dimY = 10;
ext_Arch_2C_lps = ssae_dr_abe(NB, inp_feature, past_frames, dr_params, future_frames, dimX, dimY);

%%
% Remove few samples at the start and end to remove inconsistencies for initial and last 2 frames
l = 500;
extended_speech_M2 = extended_speech_M2(l:end-l);
ext_Arch_2C_logmfe = ext_Arch_2C_logmfe(l:end-l);
ext_Arch_2C_lps = ext_Arch_2C_lps(l:end-l);
WB = WB(l:end-l);

%%
figure;    
k = 5;
a = 200; b = 50;
bx(1)=subplot(k,1,1);
specgram(NB,[],Fs16/2)
title('Spectrogram of NB speech (Fs=8Khz)')
caxis([-a,b])
ylim([0,8000])
bx(2)=subplot(k,1,2);
specgram(ext_Arch_2C_logmfe,[],16000)
caxis([-a,b])
title('Extended WB speech using SSAE dimentionality reduction with LogMFE energy as input')
bx(3)=subplot(k,1,3);
specgram(ext_Arch_2C_lps,[],16000)
caxis([-a,b])
title('Extended WB speech with log power spectrum as input')
bx(4)=subplot(k,1,4);
specgram(extended_speech_M2,[],16000)
caxis([-a,b])
title('Extended WB speech using PCA as dimentionality reduction with LogMFE energy as input')
bx(5)=subplot(k,1,5);
specgram(WB,[],16000)
caxis([-a,b])
title('Original WB speech')
linkaxes(bx,'x')

%% Objective mesures
% Define parameters 

Fs=16000; winlen=20*0.001*Fs; freq_range= 3400:10:8000; win=hanning(winlen); lpc_order=20; RS=0;

[LSD_lpc_M2, LSD_spectrum_M2, d_COSH_lpc_M2, MOS_LQO_M2] = objective_measures(WB, extended_speech_M2, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_ssae_logmfe, LSD_spectrum_ssae_logmfe, d_COSH_lpc_ssae_logmfe, MOS_LQO_ssae_logmfe] = objective_measures(WB, ext_Arch_2C_logmfe, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_ssae_lps, LSD_spectrum_ssae_lps, d_COSH_lpc_ssae_lps, MOS_LQO_ssae_lps] = objective_measures(WB, ext_Arch_2C_lps, win, winlen/2, Fs, freq_range, lpc_order, RS);
