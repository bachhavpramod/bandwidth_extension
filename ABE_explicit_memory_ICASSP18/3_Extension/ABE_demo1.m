% Script to perform Artificial bandwidth Extension (ABE).

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

%% Read speech file to be extended
path_to_file = './../Speech_files/';
file = 'A1'; % Choose any file from A1-A14 

[NB, Fs8] = audioread([path_to_file,'NB/',file,'_NB.wav']); 
[WB, Fs16] = audioread([path_to_file,'WB/',file,'_WB.wav']);

%%
global path_to_GMM;
path_to_GMM = './../2_GMM_training/existing_models/';
% path_to_GMM = './../2_GMM_training/your_models/';

%% To perform ABE using the proposed method
past_frames = 2; future_frames=2; inp_feature= 'LogMFE_zs_pca'; dimX=10; dimY=10; 
extended_speech_M2 = logmfe_lpc_abe(NB, inp_feature, past_frames, future_frames, dimX, dimY);

%% To perform ABE using the baseline method B1
past_frames = 0; future_frames=0; inp_feature= 'LogMFE'; dimX=10; dimY=10; 
extended_speech_B1 = logmfe_lpc_abe(NB, inp_feature, past_frames, future_frames, dimX, dimY);

%% To perform ABE using the baseline method B2
hlen= 2; past_frames= hlen; future_frames = hlen; inp_feature='LogMFE_mem_delta'; dimX=5; dimY=5;
extended_speech_B2 = logmfe_lpc_abe(NB, inp_feature, past_frames, future_frames, dimX, dimY);

%%
% Remove few samples at the start and end to remove inconsistencies for initial and last 2 frames
l=500;
extended_speech_M2 = extended_speech_M2(l:end-l);
extended_speech_B1 = extended_speech_B1(l:end-l);
extended_speech_B2 = extended_speech_B2(l:end-l);
WB = WB(l:end-l);

%%
figure;
k=5;
a=200;b=50;
bx(1)=subplot(k,1,1);
specgram(NB,[],Fs16/2)
title('Spectrogram of NB speech (Fs=8Khz)')
caxis([-a,b])
ylim([0,8000])
bx(2)=subplot(k,1,2);
specgram(extended_speech_M2,[],16000)
caxis([-a,b])
title('Extended WB speech using proposed method')
bx(3)=subplot(k,1,3);
specgram(extended_speech_B1,[],16000)
caxis([-a,b])
title('Extended WB speech using baseline B1')
bx(4)=subplot(k,1,4);
specgram(extended_speech_B2,[],16000)
caxis([-a,b])
title('Extended WB speech using baselineB2')
bx(5)=subplot(k,1,5);
specgram(WB,[],16000)
caxis([-a,b])
title('Original WB speech')
linkaxes(bx,'x')

%% Objective mesures
% Define parameters 

Fs=16000; winlen=20*0.001*Fs; freq_range= 3400:10:8000; win=hanning(winlen); lpc_order=20; RS=0;

[LSD_lpc_M2, LSD_spectrum_M2, d_COSH_lpc_M2, MOS_LQO_M2] = objective_measures(WB, extended_speech_M2, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_B1, LSD_spectrum_B1, d_COSH_lpc_B1, MOS_LQO_B1] = objective_measures(WB, extended_speech_B1, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_B2, LSD_spectrum_B2, d_COSH_lpc_B2, MOS_LQO_B2] = objective_measures(WB, extended_speech_B2, win, winlen/2, Fs, freq_range, lpc_order, RS);
