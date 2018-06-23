[a_WB, e_WB(:,j2_ind)]= lpc(frameWB(:,j2_ind),2*lp_order_nb);  % true WB parameters
res_WB = filter(a_WB,sqrt(e_WB(:,j2_ind)),[frameWB(:,j2_ind) ;zeros(2*lp_order_nb,1)]);
res_ext = res_WB;
RES_ext = fft(res_ext,Nfft);