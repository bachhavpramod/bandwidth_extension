% Script for FEATURE EXTRACTION for ABE.
% Written by Pramod Bachhav, June 2018
% Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

%   References:
% 
%     P.Bachhav, M. Todisco and N. Evans, "Exploiting explicit memory
%     inclusion for artificial bandwidth extension", in Proceedings of
%     ICASSP 2018, Calgary, Canada.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; close all; clear all;
addpath('./../../utilities')
path_to_speech_files='./../Speech_files/';

%% Read paths to time aligned NB and WB speech files in 'paths_to_files_NB' and 'paths_to_files_WB'

tmp = dir([path_to_speech_files,'*.wav']);
tmp1 = strrep({tmp.name},'.wav','')';    

paths_to_files_NB = strcat(path_to_speech_files,'NB/',tmp1);
paths_to_files_WB = strcat(path_to_speech_files,'WB/',tmp1);
 
%%
global params;
    
if length(paths_to_files_NB)~= length(paths_to_files_WB)
    error('Different number of files')
end

%% feat_LB and feat_HB are the features to be extracted for narrowband(NB)- and highband(HB) components respectively

%% Section 1 - Feature extraction parameters for proposed ABE system M2
feat_LB = 'LogMFE'; NumFilters_NB = 10; params.orderNB = 9; % = (Expected dimension of NB vector-1)
feat_HB = 'LPC'; params.orderHB = 9; % for proposed approach and baseline B1 

%% Section 2 - Feature extraction parameters for proposed ABE system M2
% feat_LB = 'LogMFE'; NumFilters_NB = 5; params.orderNB = 9; % = (Expected dimension of NB vector-1)
% feat_HB = 'LPC'; params.orderHB = 4;  

%%
% Uncomment one of the following lines to choose other NB feature options
% PS - Power spectrum, LPC - LP coefficients, ACF - auto-eorrelation coefficients, LogMFE - log Mel filter energy coefficients

% feat_LB='PS'; NumFilters_NB = 10; params.orderNB = 9; % = (Expected dimension of NB vector-1)
% feat_LB='LPC'; params.orderNB = 9; % = (Expected dimension of NB vector-1)
% feat_LB='ACF'; params.orderNB = 9; % = (Expected dimension of NB vector-1)

%%
diary('log.txt')  

%%
fig=0; % = 1 to check the plots, do it for only one file 
features=[];
LPF = load('./../../Filters/LPF_7700_8300.mat'); dLPF=(length(LPF.h_n)+1)/2;

%% Simulation parameters
    Fs = 16000;
    winlen_ms = 20;
    shift_ms = 10;

    ms=0.001;
    winlen = winlen_ms*Fs*ms; 
    shift = shift_ms*Fs*ms;      % hop size for hanning window
     
%% Define NB and HB frequency ranges for SLP analysis
    f1_nb = 300; f2_nb = 3400;
    f1_hb = 3400; f2_hb = 8000;
    
%% FFT processing parameters:
    params.Nfft = 2^nextpow2(2*winlen-1);    
       
%% Define mel filter bank for LogMFE features
    params.melfb_LB = melbankm(NumFilters_NB, params.Nfft, Fs, f1_nb/Fs, f2_nb/Fs);	 
    
%%
    if strcmp(feat_LB,'LPC') | strcmp(feat_LB,'ACF')
        feat_dim_NB = params.orderNB + 1; % LP coefficients + gain 
    elseif strcmp(feat_LB,'LPS')
        feat_dim_NB =[];        
    elseif strcmp(feat_LB,'LogMFE')
        feat_dim_NB=NumFilters_NB;
    end
    
    if strcmp(feat_HB,'LPC')
        feat_dim_HB = params.orderHB + 1;
    elseif strcmp(feat_HB,'LogMFE')
        feat_dim_HB = NumFilters_HB;
    end

%%  Analysis and synthesis window which satisfies the OLA constraint
    wPR = hanning(winlen,'periodic')'; K = sum(wPR)/shift; win = sqrt(wPR/K);   
    
for loop = 1:length(paths_to_files_NB)

%% Read files
    [nb, Fs8] = audioread([paths_to_files_NB{loop,1},'_NB.wav']); nb = nb(:,1)';
    [wb, Fs16] = audioread([paths_to_files_WB{loop,1},'_WB.wav']); wb = wb(:,1)';
    
%% Upsample WB signal    
    nbup = zeros(1,2*length(nb));
    nbup(1:2:length(nbup))=2*nb;
    nbup = conv(nbup,LPF.h_n); nbup=nbup(dLPF:end-dLPF+1);

    if Fs16~=16000
        error('Input file should be NB at 8khz')
    end

    [m]=min(length(nbup),length(wb));
    nbup=nbup(1:m);
    wb=wb(1:m);
    
    if fig==1
        figure
        plot(nbup)
        hold on;plot(wb,'r')
        legend('wb','swb')
    end

    Nsig =length(nbup);   
    Nframes = floor((Nsig-winlen)/shift);  % No. of frames

    feat_vect_LB=[]; 
    feat_vect_HB=[]; 
  
    for m = 0 :Nframes-1
        
        %% NB processing    
        
        index = m*shift+1:min(m*shift+winlen,length(nbup)); % indices for the mth frame
        if length(nbup(index)==0)==length(nbup(index))
            nbup(index)=nbup(index)+eps; % if frame contains only zeros
        end
        frame_nb = nbup(index).*win; 
        
        %% WB processing
        if length(wb(index)==0)==length(wb(index))
            wb(index)=wb(index)+eps;
        end
        frame_wb = wb(index).*win;  
        
%% Perform Selective linear prediction (SLP) to extract NB and HB features    
        
        [a_LB, e_LB, H_nb, ind_range_nb, R_nb, R_pos_range_nb, Rt_nb] = slp(frame_nb,[f1_nb,f2_nb],Fs16, params.orderNB,params.Nfft,0);
        [a_HB, e_HB, H_wb, ind_range_wb, R_hb, R_pos_range_hb, Rt_hb] = slp(frame_wb,[f1_hb,f2_hb],Fs16, params.orderHB,params.Nfft,0);

%% Get NB features

    if strcmp(feat_LB,'LPC')  
        feat_vect_LB(:,m+1)=[sqrt(e_LB) a_LB(2:end)];
        
    elseif strcmp(feat_LB,'LogMFE') 
        LogMFE = log(params.melfb_LB*R_nb(1:params.Nfft/2+1).');               
        feat_vect_LB(:,m+1) = LogMFE;   
        
    elseif strcmp(feat_LB,'PS')  
        feat_vect_LB(:,m+1) = R_pos_range_nb';  
        
    elseif strcmp(feat_LB,'ACF')
        tmp=real(ifft(Rt_nb.'));  

        check1=levinson(tmp,params.orderNB);
        check2 = a_LB;
        
        tmp=tmp(2:end)/tmp(1);  % take coeff for 1st lag, normalised by first coeff
        feat_vect_LB(:,m+1) = tmp(1:feat_dim_NB);        
    end
    
%% Get HB features 
    if strcmp(feat_HB,'LPC')
            feat_vect_HB(:,m+1)=[sqrt(e_HB) a_HB(2:end)];
    end                
            
    dim_LB=size(feat_vect_LB,1);
    dim_HB=size(feat_vect_HB,1);
            
%%  Plot if needed
    if fig==1
        if m==74  % random frame number to be displayed
            
            abc=levinson(real(ifft(Rt_nb.')),params.orderNB);
            abcd=a_LB;
            
            if strcmp(feat_LB,'LPC') & strcmp(feat_HB,'LPC')
                figure;
                plot((0:length(nbup(index))-1)/Fs16,nbup(index))
                hold on; plot((0:length(wb(index))-1)/Fs16,wb(index),'r')
                legend('WB speech frame','SWB speech frame')
                
            % Plots only for NB
                figure;
                ax(1)=subplot(311);
                [a_true,e_true]= lpc(frame_nb,2*params.orderNB);   % L=order
                [h_true, f_true]=freqz(sqrt(e_true),a_true,params.Nfft,'whole',Fs16);
                
                plot(f_true(1:params.Nfft/2),10*log(abs(h_true(1:params.Nfft/2))),'k');hold on;
                plot(f_true(1:params.Nfft/2),10*log(abs(H_nb((1:params.Nfft/2)))),'r');hold on;
                plot(f_true(1:params.Nfft/2),5*log(abs(R_nb((1:params.Nfft/2)))));hold on;
                % plot(f_slp(1:Nfft/2),10*log(abs(H(1:Nfft/2))),'k')
                legend('True WB spectral envelope','NB envelope via SLP','NB speech spectrum')
                
            % Plots only for HB
                ax(2)=subplot(312);
                [a_true,e_true]= lpc(frame_wb,2*params.orderHB);   % L=order
                [h_true, f_true]=freqz(sqrt(e_true),a_true,params.Nfft,'whole',Fs16);
                
                plot(f_true(1:params.Nfft/2),10*log(abs(h_true(1:params.Nfft/2))),'k');hold on;
                plot(f_true(1:params.Nfft/2),10*log(abs(H_wb((1:params.Nfft/2)))),'r');hold on;
                plot(f_true(1:params.Nfft/2),5*log(abs(R_hb((1:params.Nfft/2)))));hold on;
                % plot(f_slp(1:Nfft/2),10*log(abs(H(1:Nfft/2))),'k')
                legend('True spectral envelope','HB envelope via SLP','WB speech spectrum')
                ylabel('Magnitude (dB)')
               
                ax(3)=subplot(313);
                plot(f_true(1:params.Nfft/2),10*log(abs(h_true(1:params.Nfft/2))));hold on;
                plot(f_true(1:params.Nfft/2),10*log(abs(H_nb((1:params.Nfft/2)))),'r');hold on;
                plot(f_true(1:params.Nfft/2),10*log(abs(H_wb((1:params.Nfft/2)))),'k');hold on;
                legend('True SWB', 'f1-f2, LB', 'f1,f2, WB')
                
                figure;
                plot(f_true(1:params.Nfft/2),10*log(abs(h_true(1:params.Nfft/2))));hold on;
                plot(f_true(1:params.Nfft/2),10*log(abs(H_nb((1:params.Nfft/2)))),'r');hold on;
                plot(f_true(1:params.Nfft/2),10*log(abs(H_wb((1:params.Nfft/2)))),'k');hold on;
                legend('True WB envelope', 'NB envelope', 'HB envelope')
                
                linkaxes(ax,'xy')
                
            elseif strcmp(feat_LB,'LPS') & strcmp(feat_HB,'LPC')
                [a_true,e_true]= lpc(frame_wb,2*params.orderHB);   % L=order
                [h_true, f_true]=freqz(sqrt(e_true),a_true,params.Nfft,'whole',Fs16);
                
                figure;
                plot(f_true(1:params.Nfft/2),10*log(abs(h_true(1:params.Nfft/2))),'k');hold on;
                plot(f_true(1:params.Nfft/2),10*log(abs(H_wb((1:params.Nfft/2)))),'r');hold on;
                plot(f_true(ind_range_nb),5*log(abs(feat_vect_LB(:,m+1))));
                legend('True WB spectral envelope','HB spectral envelope','Magnitude spectrum for NB speech')
                title('All plots for NB processing')
                
%             elseif strcmp(feat_LB,'ACF')
%                 figure
%                 subplot(411)
%                 plot(tmp)
%                 title('ACF')
%                 subplot(412)
%                 plot(feat_vect_LB(:,m+1))
%                 title('ACF - first 10 coeff')
%                 subplot(413)
%                 plot(log(Pt_nb))
%                 title('Power spectra translated for ACF')
%                 subplot(414)
%                 plot(log(P_nb))
%                 title('Power spectra NB')
            end
        end     
    end
    end
           
features=[features [feat_vect_LB;feat_vect_HB]];        

disp(['Processing file ',paths_to_files_NB{loop,1}, ' : Number of frames = ',num2str(size(features,2))])  

end

diary off
pathToSave = pwd;

filename=[feat_LB,'_NB_',num2str(f1_nb),'_',num2str(f2_nb),'Hz_',feat_HB,'_','WB_',num2str(f1_hb),'_',num2str(f2_hb),'_Hz_',num2str(winlen_ms),'_',num2str(shift_ms),'ms_dim_LB=',num2str(dim_LB),'_HB=',num2str(dim_HB)];
movefile('log.txt',[filename,'.txt'])
writehtk([filename],features',dim_LB+dim_HB,9); % write one frame per row






