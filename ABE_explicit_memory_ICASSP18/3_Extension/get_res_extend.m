% Get residual from upsampled NB signal and extended WB spectral envelope
    res_NBup = filter(a_ext, sqrt(e_ext(:,j2_ind)),[frameNBup(:,j2_ind);zeros(lp_order_wb,1)]);
    n=(0:length(res_NBup)-1)';

if SM==1
% Spectral mirroring using translation
    res_ext = res_NBup.*cos(2*pi*8000*n/Fs16);   % cos(2*pi*f*t)=cos(2*pi*f*n*Ts)
    res_ext = res_ext + res_NBup;
elseif translation_freq~=0
    res_ext = res_NBup.*cos(2*pi*translation_freq*n/Fs16);   % cos(2*pi*f*t)=cos(2*pi*f*n*Ts)
    res_ext_hb = conv(res_ext,HPF.h_n); 
    res_ext_hb = res_ext_hb(dHPF:end-dHPF+1)';
    res_ext = 2*res_ext_hb + res_NBup;
end

RES_ext = fft(res_ext,Nfft);
