% Usage example of SWBE_LPAS function for SUPER-WIDE BADWIDTH EXTENSION USING LINEAR PREDICTION BASED ANALYSIS-SYNTHESIS 
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   References:
% 
%     P.Bachhav, M. Todisco and N. Evans, "Efficient super-wide bandwidth extension
%     using linear prediction based analysis synthesis", in Proceedings of
%     ICASSP 2018, Calgary, Canada.
%
%     Users are requested to cite the above paper in papers which report 
%     results obtained with this function

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
addpath('./../utilities')

%% Parameters
Fs_swb = 32000;
Fs_wb = 16000;

ms = 0.001;
winlen_swb = 25*ms*Fs_swb; 
LP_order_wb = 16;
Nfft = 1024;
gain = 1; % GAIN value can be adjusted to check the effect of energy on speech quality
% Set to 1 for the proposed method and the basline used in the paper

%% Load filters
LPF = load('./../Filters/LPF_7700_8300.mat'); dLPF=(length(LPF.h_n)+1)/2;
HPF = load('./../Filters/HPF_7700_8300.mat'); dHPF=(length(HPF.h_n)+1)/2;
BPF = load('./../Filters/BPF_4000_8000.mat'); dBPF=(length(BPF.h_n)+1)/2;

filters = [];
filters.LPF = LPF;
filters.HPF = HPF;
filters.BPF = BPF;

%% Read speech files
[AMRWB Fs16] = audioread('./speech_files/bdl_arctic_a0001_AMR2.wav');
[EVS Fs32] = audioread('./speech_files/bdl_arctic_a0001_EVS13200_dtx_rf.wav');
[SWB Fs32] = audioread('./speech_files/bdl_arctic_a0001_SWB.wav');

if size(AMRWB,2)==1
    AMRWB = AMRWB'; % make the file as a row vector
end      
if Fs16~=16000
    error('Input file should be WB at 16kHz')
end

extended_SWB_LPAS = SWBE_LPAS(AMRWB, LP_order_wb, Nfft, winlen_swb, gain, filters);
extended_SWB_EHBE = SWBE_EHBE(AMRWB, gain, filters);

%% Bandlimit extended speech file to 15kHz
LPFend = load('./../Filters/LPF_14100_14400_32k.mat');
dLPFend = (length(LPFend.h_n)+1)/2;
extended_SWB_LPAS = conv(extended_SWB_LPAS,LPFend.h_n); extended_SWB_LPAS = extended_SWB_LPAS(dLPFend:end-dLPFend+1);
extended_SWB_EHBE = conv(extended_SWB_EHBE,LPFend.h_n); extended_SWB_EHBE = extended_SWB_EHBE(dLPFend:end-dLPFend+1);

%%
% Remove few samples at the start and end to remove inconsistencies for initial and last 2 frames
l = 3000;
AMRWB = AMRWB(l/2:end-l/2);
extended_SWB_LPAS = extended_SWB_LPAS(l:end-l);
extended_SWB_EHBE = extended_SWB_EHBE(l:end-l);
EVS = EVS(l:end-l);
SWB = SWB(l:end-l);

%%
figure
lim=140;lim1=20;
bx(1)=subplot(5,1,1);
specgram(AMRWB,[],Fs16)
ylim([0,16000])
title('AMR-WB speech')
caxis([-lim,lim1])
bx(2)=subplot(5,1,2);
specgram(extended_SWB_LPAS,[],Fs32)
caxis([-lim,lim1])
title('Extended SWB speech - LPAS (Proposed)')
bx(3)=subplot(5,1,3);
specgram(extended_SWB_LPAS,[],Fs32)
caxis([-lim,lim1])
title('Extended SWB speech - EHBE (Baseline)')
bx(4)=subplot(5,1,4);
specgram(EVS,[],Fs32)
caxis([-lim,lim1])
title('EVS processed speech')
bx(5)=subplot(5,1,5);
specgram(EVS,[],Fs32)
caxis([-lim,lim1])
title('Original SWB speech')
linkaxes(bx,'x')

%% Objective mesures
% Define parameters 

Fs = 32000; winlen = 20*0.001*Fs; freq_range= 50:10:14000; win=hanning(winlen); lpc_order=20; RS=0;

[LSD_lpc_LPAS, LSD_spectrum_LPAS, d_COSH_lpc_LPAS, ~] = objective_measures(SWB, extended_SWB_LPAS, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_EHBE, LSD_spectrum_EHBE, d_COSH_lpc_EHBE, ~] = objective_measures(SWB, extended_SWB_EHBE, win, winlen/2, Fs, freq_range, lpc_order, RS);
[LSD_lpc_EVS, LSD_spectrum_EVS, d_COSH_lpc_EVS, ~] = objective_measures(SWB, EVS, win, winlen/2, Fs, freq_range, lpc_order, RS);

