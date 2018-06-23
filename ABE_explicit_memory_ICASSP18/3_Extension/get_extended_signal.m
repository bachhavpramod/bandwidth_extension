frame_extended = RES_ext.* H;               % freq domain multiplication
frame_extended = real(ifft(frame_extended));         % inverse transform

% Multiply extended speech frame by synthesis window for perfect reconstruction using OLA 
frame_extended = frame_extended'.*[win_wb,zeros(1,Nfft-length(win_wb))];  
outindex = (j2_ind-1)*shift_wb+1:((j2_ind-1)*shift_wb+Nfft);

extended_speech(outindex) = extended_speech(outindex) + frame_extended; % overlap add

%%
% True NB LP coefficients and LP gain    
[a_nb,e_nb] = lpc(frameNB, lp_order_nb); 
