% Get NB excitation (Fs=8kHz)
[a_nb,e_nb]=lpc(frameNB, lp_order_nb);  % true NB parameters
res_NB=filter(a_nb,sqrt(e_nb),frameNB); % as a_NB is obt thru up NB signal

% Get NB excitation (Fs=16kHz) using  extended WB envelope and upsampled NB signal
res_NBup = filter(a_ext,sqrt(e_ext(:,plot_frame)),[frameNBup(:,plot_frame);zeros(lp_order_wb,1)]);

% Get original WB excitation
[a_WB, e_WB(:,plot_frame)]=lpc(frameWB(:,plot_frame), lp_order_wb);  % true WB parameters
res_WB = filter(a_WB,sqrt(e_WB(:,plot_frame)),[frameWB(:,plot_frame);zeros(lp_order_wb,1)]);

% Take FFT
[RES_WB f]=freqz(res_WB,1,Nfft,'whole',Fs16);
[RES_WBup f]=freqz(res_NBup,1,Nfft,'whole',Fs16);
[FRAMEWBup f]=freqz(frameNBup(:,plot_frame),1,Nfft,'whole',Fs16);
[RES_NB f11]=freqz(res_NB,1,Nfft/2,'whole',Fs16/2);

figure;
ax1(1)=subplot(411);
plot(f11,20*log(abs(RES_NB)),'r');
xlim([0,Fs16])
title('FFT - true NB excitation (Fs=8kHz)')
ax1(2)=subplot(412);
plot(f,20*log(abs(RES_WBup)),'k'); hold on;
title('WB excitation obtained using extended WB envelope and upsampled NB signal')
ylabel('Magnitude (dB)')
ax1(3)=subplot(413);
plot(f,20*log(abs(RES_ext)))
title('Extended excitation (using spectral translation at 6800Hz) ')
ax1(4)=subplot(414);
plot(f,20*log(abs(RES_WB)),'r')
linkaxes(ax1,'xy')
title('true WB excitation')
xlabel('Frequency (Hz)')

figure;
plot(f,20*log(abs(H_ext_wb)));hold on;
plot(f,20*log(abs(H)),'k');hold on;
plot(f,20*log(abs(H_NBup)),'r');
plot(f,10*log(P_nb),'g');hold on;
legend('Extended envelope before Levinson-Durbin', 'Extended envelope after Levinson-Durbin', ...
                'Envelope of upsapled NB speech frame', 'Power spectrum of upsampled speech frame')
ylabel('Magnitude (dB)')
xlabel('Frequency (Hz)')
title('Frequency responses of spectral envelopes')

figure;
plot(f,20*log(abs(H_nb(:,plot_frame))));hold on;
plot(f,20*log(abs(H_hb)),'r');hold on;
plot(f,20*log(abs(H_WB)),'g');hold on;
plot(f,20*log(abs(H)),'k');hold on;
legend('NB spectral envelope','Estimated HB envelope','True WB envelope','Estimated WB envelope')
title('Frequency responses of spectral envelopes')

