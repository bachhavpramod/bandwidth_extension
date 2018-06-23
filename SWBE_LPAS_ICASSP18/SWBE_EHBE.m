function speech_extended = SWBE_EHBE(speech_wb, gain, filters)

% EFFICIENT HIGH-FREQUENCY BANDWIDTH EXTENSION OF MUSIC AND SPEECH (SWBE_EHBE)
% Time-domain implementation
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   Input parameters:
%         speech_wb        : input wideband (WB) signal
%         lp_order_wb      : sampling frequency
%         gain             : gain to adjust energy of highband (HB) components
%         filters          : structure containing filters
%
%   Output parameters:
%         speech_extended  : extended super-wide band (SWB) signal
%
%   References:
% 
%     E. Larsen, R. M. Aarts, and M. Danessis, “Efficient highfrequency
%     bandwidth extension of music and speech,” in Audio
%     Engineering Society Convention 112. Audio Engineering
%     Society, 2002.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Filter delays
dLPF=(length(filters.LPF.h_n)+1)/2;
dHPF=(length(filters.HPF.h_n)+1)/2;
dBPF=(length(filters.BPF.h_n)+1)/2;

%% Upsample input WB signal
UpWB = zeros(1,2*length(speech_wb));
UpWB(1:2:length(UpWB)) = 2*speech_wb;
UpWB = conv(UpWB, filters.LPF.h_n); 
UpWB = UpWB(dLPF:end-dLPF+1);

%% Extract highest octave from the upsampled WB signal
OctaveUpWB = conv(filters.BPF.h_n,UpWB);
OctaveUpWB=OctaveUpWB(dBPF:end-dBPF+1);

%% abs
UpOctaveWBabs = abs(OctaveUpWB);

%% Extract HB speech
UpOctaveWBabsHB = conv(filters.HPF.h_n,UpOctaveWBabs);
UpOctaveWBabsHB = UpOctaveWBabsHB(dHPF:end-dHPF+1);

%% Synthesize extended speech
speech_extended = UpWB + gain*UpOctaveWBabsHB;

end


     