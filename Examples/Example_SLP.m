% Usage example of slp function for SELECTIVE LINEAR PREDICTION 

% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   References:
% 
%     J. Markel and A. Gray, Linear prediction of speech. Springer
%     Science & Business Media, 2013, vol. 12.
%     
%     Refer section 6.4.1 (page no. 148) for more details 

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

[sig Fs] = audioread('file.wav');
L1=12801; L2=L1+320; 

frame = sig(L1:L2);    

M=20; % SLP order 
N=length(frame);

% Frequency rangre to perform SLP
% Case 1 
f_range=[3400,8000]; LP_order = 2*M; % LP order for complete spectrum

% Case 2
% f_range=[0,8000]; LP_order = M; % LP order for complete spectrum

Nfft=2^nextpow2(length(frame)+M+1);  % Nfft should be > N+M and next to the power of 2

append =0;
[a_slp, e_slp, H, ind_range, P]= slp(frame,f_range,Fs,M,Nfft, append);

[a_true,e_true] = lpc(frame, LP_order);
[h_true, f_true]=freqz(sqrt(e_true),a_true,Nfft,'whole',Fs);

figure;
plot(f_true(1:Nfft/2),20*log(abs(h_true(1:Nfft/2))),'k');hold on;
plot(f_true(1:Nfft/2),20*log(abs(H((1:Nfft/2)))),'r-*');hold on;
plot(f_true(1:Nfft/2),10*log(abs(P((1:Nfft/2)))));hold on;
legend('Complete spectral envelope','Envelope for frequency range f1-f2 obtained via SLP','speech spectrum')
ylabel('Magnitude (dB)')
xlabel('Frequency')