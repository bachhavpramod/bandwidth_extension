function I=eVAD(framedspeech)

% Voice activity detection (VAD)

% function I=eVAD(speech ,frameLength_inSamples,frameShift_inSamples)

% Reference:
%   T Kinnunen, H Li. An overview of text-independent speaker
%   recognition: From features to supervectors.
%   Speech Communication 52 (1), 12-40, 2010

% x_framed = buffer(speech, frameLength_inSamples, frameShift_inSamples);
% framedspeech = x_framed(:,2:end-1);
% E=20*log10(std(framedspeech,0,1)+eps);
% maxl=max(E);
% I=(E>maxl-30) & (E>-55);

% Edited by Pramod Bachhav 
% done acd to enframe
E = 20*log10(std(framedspeech,0,2)+eps);   % 2- std of each row, 1- std of each column
maxl=max(E);
I=(E>maxl-30) & (E>-55);
end