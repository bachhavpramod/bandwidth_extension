function [a_slp, e_slp, H, ind_range, R, R_pos_range, Rt] = ...
    slp( speech_frame, f_range, Fs, M, Nfft, append)

% Function for SELECTIVE LINEAR PREDICTION (SLP)
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   Input parameters:
%      speech_frame  : input speech frame
%      f_range       : frequency range over which SLP is to be performed
%      Fs            : Sampling frequency in Hz
%      M             : filter order
%      Nfft          : FFT order
%      append        :
%
%   Output parameters:
%      a_slp         : Linear prediction (LP) envelope obtained via SLP for frequency range f_range
%      e_slp         : corresponding residual error (or excitation)
%      H             : A sequence of zeros (of length Nfft) with frequency components of signal present in f_range
%                      Used for extension and plots 
%      ind_range     : bins corresponding to f_range
%      other parameters can be returned for plots

%   References:
% 
%     J. Markel and A. Gray, Linear prediction of speech. Springer
%     Science & Business Media, 2013, vol. 12.
%     
%     Refer section 6.4.1 (page no. 148) for more details 
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

%% Step 1: Compute the data sequence spectrum.
if size(speech_frame,1)> size(speech_frame,2)
    speech_frame  = speech_frame'; % tun into a row vector
end

N=length(speech_frame);
R=(abs(fft(speech_frame,Nfft))).^2/N;  % Eq. 6.31 & 6.32
% divided by N to get a biased estimate so that the gain obtained via levinson matches 
% with LP gain obtained using 'lpc' command 

%% Step 2: Create a translated spectrum.
f1 = f_range(1);
f2 = f_range(2);
I = Nfft; % = 2^IP

% k1 and k2 are the bins correspinding to f1 and f2 Hz
% For even I, 0 < k1 < I/2 and 
%             k1 < k2;
k1 = round(f1*I/Fs);
k2 = round(f2*I/Fs);

ind_range = k1+1:k2+1; % range of bins corresponding to f_range
R_pos_range = R(ind_range); 

l = k2-k1; 
k = 0:l;
pos_ind=(k1 + k); % i.e. indices for translated spectrum, k' = k1 + k where k= 0,...,l

temp=[R(pos_ind+1)]; % +1 for matlab index - pos_ind+1 and ind_range are the same

Rt=[temp, fliplr(temp(2:end-1))]; % Eq. 6.33a and 6.33b
% We choose length of resultant translated spectrum, L = 2*l here
% Spectrum Rt can be thought of as the result of analyzing some data sequence with an L~point DFT.

L = 2*l;
L_pos_hp = L/2+1; % number of bins corresponding to positive freqs 

%% Steps 3 and 4
% Perform Levison-Durbin on translated spectrum to get corresponding spectral envelope and LP gain
[a_slp,e_slp] = levinson(real(ifft(Rt)),M);

%%
[h_slp] = freqz(sqrt(e_slp),a_slp,length(Rt),'whole',Fs);
H=zeros(1,Nfft/2+1); % for even length fft
H(pos_ind+1)=h_slp(1:L_pos_hp);
xyz=ind_range(1);
    if append==1
        H(1:xyz-1)= sqrt(R(1:xyz-1));
    end
H =[H, fliplr(H(2:end-1))];
