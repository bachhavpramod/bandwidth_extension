% Demo for Artificial bandwidth Extension (ABE).

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
file = 'A8'; % Choose any file from A1-A14 

[NB, Fs8] = audioread([path_to_file,'NB/',file,'_NB.wav']); 
[WB, Fs16] = audioread([path_to_file,'WB/',file,'_WB.wav']);

%%
global path_to_GMM;
path_to_GMM = './../2_GMM_training/existing_models/';
% path_to_GMM = './../2_GMM_training/your_models/';

%% To perform ABE using the proposed method
past_frames = 2; future_frames=2; inp_feature= 'LogMFE_zs_pca'; dimX=10; dimY=10; 
extended_speech_M2 = logmfe_lpc_abe(NB, inp_feature, past_frames, future_frames, dimX, dimY, WB);