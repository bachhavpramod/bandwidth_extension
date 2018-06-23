%% Get HB power spectra (these all things are for eb)
  
l_h_hb = 2*length(pos_ind)-1;
temp = freqz(g_hb, a_hb, l_h_hb,'whole',Fs16);

L_pos_hp= L/2+1; % length of bins corresponding to positive freqs

% Translate spectrum 'temp' to HB, get positive side of the spectrum
H_hb = zeros(1, Nfft/2+1);
H_hb(pos_ind+1) = temp(1:L_pos_hp);

% Create the complete HB spectrum
H_hb=[H_hb,fliplr(H_hb(2:end-1))];

%% Combined NB and HB spectra to get combine power spectra
H_ext_wb = H_hb + H_nb(:, j2_ind)';

P_ext = abs(H_ext_wb).^2;
[a_ext,e_ext(:,j2_ind)] = levinson(real(ifft(P_ext)), lp_order_wb);

if ~isstable(1,a_ext)
    error('unstable filter')
end

% Frequency response of the extended WB spectral envelope after Levinson-Durbin
H = freqz(sqrt(e_ext(:,j2_ind)),a_ext,Nfft,'whole',Fs16);

%% For plots
if fig == 1
    % True WB LP coefficients and LP gain
    [a_WB,e_WB(:,j2_ind)] = lpc(frameWB(:,j2_ind), lp_order_wb);
end