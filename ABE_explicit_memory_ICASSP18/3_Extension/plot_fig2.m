figure;
plot(sqrt(e_ext));
hold on;
plot(sqrt(e_WB),'r');
legend('LP gain of estimated WB spectral envelope','Gain for true WB envelope')
title('Comparison of gains')

figure
ax(1)=subplot(311);
plot(resample(NB,2,1))
title('Upsampled NB speech (Fs=16kHz)')
ax(2)=subplot(312);
plot(extended_speech,'r')
title('Extended WB speech')
ax(3)=subplot(313);
plot(WB,'g')
title('Original WB speech')
linkaxes(ax,'xy')

figure;    
k=3;
a=200;b=50;
bx(1)=subplot(k,1,1);
specgram(NB,[],Fs16/2)
title('Spectrogram of NB speech (Fs=8Khz)')
caxis([-a,b])
ylim([0,8000])
bx(2)=subplot(k,1,2);
specgram(extended_speech,[],16000)
caxis([-a,b])
title('Extended WB speech')
bx(3)=subplot(k,1,3);
specgram(WB,[],16000)
caxis([-a,b])
title('Original WB speech')
linkaxes(bx,'x')