FRAME_WB = fft(frameWB(:,j2_ind), Nfft);
P_wb = (abs(FRAME_WB)).^2/length(frameWB(:,j2_ind));
P_eb_tran = P_wb(pos_ind+1);
P_eb_tran = [P_eb_tran; fliplr(P_eb_tran(2:end-1)')'];
% fliplr returns same vector if its column 
[abc def] = levinson(real(ifft(P_eb_tran)), lp_order_nb);
abc = abc(:);