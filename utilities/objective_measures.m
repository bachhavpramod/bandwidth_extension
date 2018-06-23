function [LSD_lpc, LSD_spectrum, d_COSH_lpc, MOS_LQO] = objective_measures(orig, ext, win, shift, Fs, Freq_range, lpc_order, RS)

% This function returns the various objective measures used for evaluation
% of arificial bandwidth extension (ABE) methods

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   Input parameters:
%         orig            : original speech file
%         ext             : extended (or processed) speech file
%         win             : window used for enframing the signal
%         shift           : shift in samples
%         Fs              : sampling frequency
%         Freq_reange     : frequency range over which spectral distortion
%                           measures to be calculated
%         lpc_order       : order for linear prediction (LP) analysis
%         RS              : 1 to remove silences before calculating
%                           distortion measures
% 
%   Output parameters:
%         LSD_lpc         : Log spectral distortion using spectral estimation from LP coefficients
%         LS_spectrum     : Log spectral distortion from power spectrum directly
%         d_COSH_lpc      : COSH measure, a smmetrized version of Itakura-Saito (IS) distortion
%                               above 3 measures are calculated for each frame and thenaveraged
%         MOS_LQO         : a WB extension of PESQ (P.862.2) which gives objective estimate of mean-opinion scores (MOS)
% 
%   References:
% 
%     R. M. Gray, A. Buzo, A. H. Gray, Jr. and Y. Matsuyama, “Distortion
%     measures for speech processing”, IEEE Transactions on Acoustics, 
%     Speech and Signal Processing, 1980.
% 
%   Acknowledgement:
% 
%     Code for log spectral distortion using LP coefficients was taken and modified from 
%     https://brage.bibsys.no/xmlui/handle/11250/2370992
% 
%     The function pesq_mex_vec is taken and modified from from 
%     'https://www.soundzones.com/2015/11/10/a-fast-matlab-executable-mex-compilation-of-the-pesq-measure/'  
%     It allows to read a MOS-LQO value from using 'pesq.exe'
%     Thanks to Jacob Donley. Please refer to above link for more details.
%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) 2016 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Convert to row vectors
if size(orig,2)==1
    orig=orig';
end
if size(ext,2)==1
    ext=ext';
end

% Normalise the signals
orig = orig/max(abs(orig));
ext = ext/max(abs(ext));

% Equal the lengths
m = min([length(orig) length(ext)]);
orig = orig(1:m);
ext = ext(1:m);

% Enframe signals, count number of frames.
[frames_orig t] = enframe(orig, win, shift);  
frames_ext = enframe(ext, win, shift); 

num_frames = size(frames_orig, 1);

% Calculate LP coefficients
[lpc_orig, g_orig] = lpc(frames_orig', lpc_order);
[lpc_ext, g_ext]= lpc(frames_ext', lpc_order);

% Intialization 
lsd_lpc = zeros(num_frames, 1); % Log spectral distortion using LP coefficients 
lsd_spectrum = zeros(num_frames, 1); % Log spectral distortion from power spectrum directly
d_cosh_lpc= zeros(num_frames, 1); % Ikatura-Saito distortion

if RS==1
    I = eVAD(frames_orig);
else
    I = ones(num_frames,1);
end

J = I;
if length(I)~=size(frames_ext,1)
    error('Check I and number of frames')
end

%% Compute RMS-LSD for each set of LP-filters.

for i_frame = 1:num_frames
    
    if length(find(frames_ext(i_frame,:)~=0))==0
       % to get new VAD to neglect the frames of exended signal with zeros, if any 
       J(i_frame)=0;
    end     
    
    if I(i_frame)~=0
        
    %% 1 LSD based on LPC envelopes
    
        H_orig = freqz(1, lpc_orig(i_frame, :), Freq_range, Fs);    
        H_ext = freqz(1, lpc_ext(i_frame, :), Freq_range, Fs);

        P_orig = 20*log10(sqrt(g_orig(i_frame)+eps)*abs(H_orig));
        P_ext = 20*log10(sqrt(g_ext(i_frame)+eps)*abs(H_ext));
        lsd_lpc(i_frame) = sqrt( mean( ( P_orig - P_ext ).^2 ) );
        
        %% 2 dCosh based on LPC envelopes
        % gives same result for log or log10
        a =  mean( (abs(H_orig).^2./abs(H_ext).^2) - log10((abs(H_orig).^2./abs(H_ext).^2 )+eps) -1 ) ;
        b =  mean( (abs(H_ext).^2./abs(H_orig).^2) - log10((abs(H_ext).^2./abs(H_orig).^2 )+eps) -1 ) ;
        d_cosh_lpc(i_frame)= (a + b)/2;

        %% 3 LSD based on spectrum
        fft_orig= freqz(frames_orig(i_frame, :) , 1 , Freq_range, Fs);    
        fft_ext= freqz(frames_ext(i_frame, :) , 1 , Freq_range, Fs);    
        
        P_orig_spectrum = 20*log10(abs(fft_orig)+eps);
        P_ext_spectrum = 20*log10(abs(fft_ext)+eps);
        
        % To check if there is any NaN or 
        tp1 = find(isnan(P_orig_spectrum));
        tp2 = find(isnan(P_ext_spectrum));
        tp3 = find(isinf(P_orig_spectrum));
        tp4 = find(isinf(P_ext_spectrum));
        
        if  ~isempty(tp1) | ~isempty(tp2) | ~isempty(tp3) | ~isempty(tp4)
            P_orig_spectrum=0;
            P_ext_spectrum=0;
        end
        lsd_spectrum(i_frame) = sqrt( mean( ( P_orig_spectrum - P_ext_spectrum ).^2 ) );
        
    end
end
l = length(find(J~=0));

lsd_lpc_new = lsd_lpc(find(J~=0));
LSD_lpc = mean(lsd_lpc_new); % neglect non-speech frames for average

lsd_spectrum_new = lsd_spectrum(find(J~=0));
LSD_spectrum = mean(lsd_spectrum_new);

d_is_new = d_cosh_lpc(find(J~=0));
d_COSH_lpc = mean(d_is_new);

[MOS_LQO] = pesq_mex_vec(orig,ext,16000);

