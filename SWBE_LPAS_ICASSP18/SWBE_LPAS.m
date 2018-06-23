function speech_extended = SWBE_LPAS( speech_wb, lp_order_wb, Nfft, winlen_swb, gain, filters)

% SUPER-WIDE BADWIDTH EXTENSION USING LINEAR PREDICTION BASED ANALYSIS-SYNTHESIS (SWBE_LPAS)
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   Input parameters:
%         speech_wb        : input wideband (WB) signal
%         lp_order_wb      : sampling frequency
%         Nfft             : FFT order
%         winlen_swb       : window length in samples
%         gain             : gain to adjust energy of highband (HB) components
%         filters          : structure containing filters
%
%   Output parameters:
%         speech_extended  : extended super-wide band (SWB) signal
%
%   References:
% 
%     P.Bachhav, M. Todisco and N. Evans, "Efficient super-wide bandwidth extension
%     using linear prediction based analysis synthesis", in Proceedings of
%     ICASSP 2018, Calgary, Canada.
%
%     Users are REQUESTED to cite the above paper if this function is used. 
%  
%   Acknowledgements:
%     
%     The basic code for overlap-add processing is taken and modified from:
%     https://www.dsprelated.com/freebooks/sasp/Overlap_Add_OLA_STFT_Processing.html
% 
%     Analysis and synthesis window selection is done from ref [1] given at 
%     https://fr.mathworks.com/matlabcentral/fileexchange/45577-inverse-short-time-fourier-transformation--istft--with-matlab-implementation
%     Thanks to  Hristo Zhivomirov for list of nice references. 
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

%% Parameters

l_wb = length(speech_wb);
l_swb = 2* l_wb;     % signal length in samples

shift_swb = winlen_swb/2;  % 50% hop size

wPR = hanning(winlen_swb,'periodic')';K=sum(wPR)/shift_swb; 
win_swb = sqrt(wPR/K); % Hann window with OLA constraint
wPR = hanning(winlen_swb/2,'periodic')';K=sum(wPR)/(shift_swb/2); 
win_wb = sqrt(wPR/K);

Nframes = 1+floor((l_swb-winlen_swb)/shift_swb);  % number of complete frames

%% Filter delays
dLPF = (length(filters.LPF.h_n)+1)/2;
dHPF = (length(filters.HPF.h_n)+1)/2;

%% Carry out the overlap-add FFT processing:
speech_WB_upsampled = zeros(1,l_swb + Nfft);
speech_HB = zeros(1,l_swb + Nfft);

for m = 0 :Nframes-1
    
%% Input WB signal framing 
    index_wb = m*shift_swb/2+1:min(m*shift_swb/2+winlen_swb/2,l_swb);
    seg = speech_wb(index_wb);
    frame_wb = seg.*win_wb;     
    
    if length(frame_wb==0)==length(frame_wb)
        frame_wb=frame_wb+eps; % if the speech frame contains only zeros
    end

%% Get WB LP spectral envelope and residual/excitation 
    [a_wb g_wb]= lpc(frame_wb,lp_order_wb);   
    [res_wb]=filter(a_wb,1,[frame_wb,zeros(1,lp_order_wb)]);   

    H = fft(impz(sqrt(g_wb), a_wb, Nfft)');  % WB spectral envlope frequency response
   
%% Excitation extension via zero insertion
    res_ext = zeros(1,2*length(res_wb));res_ext(1:2:length(res_ext))=res_wb; 
    res_ext = res_ext./sqrt(sum(res_ext.^2)./(length(frame_wb)));
 
    RES_ext = fft(res_ext, Nfft);   
    
%% Resynthesize the HB speech              
    temp = RES_ext.* H;               
    FRAME_hb = temp.*fft( filters.HPF.h_n,Nfft); % apply HPF to get highband 
    frame_hb = real(ifft(FRAME_hb));      
    
    frame_hb=frame_hb.*[win_swb,zeros(1,Nfft-length(win_swb))];  
        
    if find(isnan(frame_hb)==1)
        disp('synthesised frame has NaN values')
        break
    end
    
    index_swb = m*shift_swb+1:(m*shift_swb+Nfft);
    speech_HB(index_swb) = speech_HB(index_swb) + frame_hb; % overlap add
        
%% Get NB components from upsampled WB signal

    frame16up=zeros(1,2*length(seg)); frame16up(1:2:length(frame16up))=2*frame_wb;
    frame16up=ifft(fft(frame16up,Nfft).*fft( filters.LPF.h_n,Nfft)).*[win_swb,zeros(1,Nfft-length(win_swb))];  
    % this is equivalent to upsampling the input WB signal to 32kHz
    speech_WB_upsampled(index_swb) = speech_WB_upsampled(index_swb) + frame16up; % overlap add
           
end

%% adjust delays
speech_WB_upsampled =[speech_WB_upsampled(dLPF:end) zeros(1,dLPF-1)];
speech_HB =[speech_HB(dHPF:end) zeros(1,dHPF-1)]; 

%% combine HB and NB components 
speech_extended = gain*speech_HB + speech_WB_upsampled;
speech_extended(end-Nfft+1:end)=[]; 

end