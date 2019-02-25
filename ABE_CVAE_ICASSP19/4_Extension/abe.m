function extended_speech = abe(NB, GMM_file, inp_feature, dr_params, past_frames, future_frames, dimX, reduced_dim, dimY, WB)

paths();

if nargin <10; fig = 0; else; fig = 1; end
plot_frame = 87; % frame number for the plots

%%
mu = [];
stdev = [];

%%

% path_to_GMM_models = './../3_GMM_training/';
% 
% comp = 128;
% suffix = ['_zs=',num2str(dr_params.zs),'_GMM_',num2str(comp),'.mat'];
% GMM_file = [config, suffix];

load(GMM_file);
disp(['Performing ABE using GMM model = ', GMM_file])  

%% Parameters definition need for function GMMR which performs regression using GMM 

comp_means = obj.mu';   % means for every component (every model) % (dimX+dimY) x NumOfComp
comp_variance = obj.Sigma;
apriory_prob = obj.PComponents;
gmmr = offline_param(apriory_prob,comp_means,comp_variance, reduced_dim);

%% Load filters
LPF = load('./../../Filters/LPF_7700_8300.mat'); dLPF=(length(LPF.h_n)+1)/2;
HPF = load('./../../Filters/HPF_3300_3400.mat'); dHPF=(length(HPF.h_n)+1)/2;

%% Parameters
    Fs8=8000; 
    Fs16=16000;
    
    wlen_nb = 20*0.001*Fs8; 
    shift_nb = wlen_nb/2;
    
    wlen_wb = 20*0.001*Fs16; 
    shift_wb = wlen_wb/2;
      
    lp_order_nb = dimY-1; % for LB=16, 1 is energy coeff, 14 are LPCs. We get X in this code by concatenating gain with LPCs    
    lp_order_wb=2*lp_order_nb+1;

    wPR=hanning(wlen_wb,'periodic')'; K=sum(wPR)/shift_wb; win_wb=sqrt(wPR/K); % Anaysis and synth wimdow which satisfies the constraint

%% Get means and variances obtained from training data 
    mu_x = mu(1:dimX);  mu_y = mu(dimX+1:end);
    stdev_x = stdev(1:dimX);  stdev_y = stdev(dimX+1:end);

%% Parameters
Nfft = 2^nextpow2(2*wlen_wb-1);
Nsig = 2*length(NB);

% Parameters form excitation extension using spectral translation
SM = 0; translation_freq = 6800; 
% For spectral mirroring , use SM=1
% Refer, get_res_extend_mem for more details

% Define parameters to perform selctive linear prediction (SLP) on highband (HB)- 3.4-8 kHz
f1 = 3400; f2=8000;
I = Nfft;
l1 = round(f1*I/Fs16); l2=round(f2*I/Fs16);
ind_range_hb = l1+1:l2+1;
l=l2-l1; L=2*l;
k=0:l; 
pos_ind = (k+l1); % bins corresponding to positive freuquencies only
    
%% Initialzation
a_nb=[];e_nb=[];
H_nb=[];H_hb=[];H=[];
a_ext=[];e_ext=[];
a_NBup=[];

frameNBup =[];
frameWB = [];
inp=[]; 

extended_speech = zeros(1,Nsig + Nfft);
Nframes = floor((Nsig-wlen_wb)/shift_wb);
res1 = zeros(1,Nsig + Nfft);
res2 = zeros(1,Nsig + Nfft);
res_orig_NB = zeros(1,Nsig + Nfft);

X_conc = zeros((past_frames+future_frames+1)*dimX,Nframes);
X_normalised = zeros(reduced_dim,Nframes);    
Y_normalised = zeros(dimY,Nframes);
Y = zeros(dimY,Nframes);
    
buffer =[];  
tmp =[];
j1_ind = past_frames+1; % j1_ind and j2_ind represents index for the frame for which ABE is performed  
j2_ind = j1_ind;

% h = 0.0;
% msg = waitbar(h,'Please wait...');

for frame = 1: Nframes

    indexNB = (frame-1)*shift_nb+1:min((frame-1)*shift_nb+wlen_nb,length(NB)); % indices for the mth frame
    
    if fig==1
        indexWB = (frame-1)*shift_wb+1:min((frame-1)*shift_wb+wlen_wb,length(WB)); % indices for the mth frame    
    end
    
    frameNB= NB(indexNB);
    if length(find(frameNB==0)) == length(frameNB)
       frameNB = frameNB+eps;
    end    

%% Upsample NB speech frame    
    frameNBup1=zeros(1,wlen_wb)';
    frameNBup1(1:2:length(frameNBup1))=2*frameNB;
    frameNBup1=conv(frameNBup1,LPF.h_n); frameNBup1=frameNBup1(dLPF:end-dLPF+1);
    frameNBup(:,frame)=frameNBup1.*win_wb';
    
%    frameNBup=resample(frameNB,2,1); % OR

    if fig==1
        frameWB1=WB(indexWB);
        if length(frameWB1==0)==length(frameWB1)
            frameWB1=frameWB1+eps;
        end    
        frameWB(:,frame) = frameWB1.*win_wb';  
        
%     To check perfect reconstruction, uncomment following line - can be used to debug the code
%     frameNBup = frameWB; 
    end
       
%% Perform SLP on NB band to get NB LP coefficients and LP gain

    SIG_NB = fft(frameNBup(:,frame),Nfft);
    P_nb = (abs(SIG_NB)).^2/length(frameNBup); 

% Perform SLP    
    [a_slp, e_slp, H_nb(:,frame), ind_range, ~, R_pos_range ]= slp(frameNBup(:,frame)',...
                                                                        [300,3400], Fs16, lp_order_nb,Nfft, 1);

    if strcmp(inp_feature,'LogMFE')
% Compute LogMFE features
        LogMFE(:,frame) = log(melfb_LB*P_nb(1:(Nfft/2)+1));          
    elseif strcmp(inp_feature,'LPS')
% Compute LPS
        LogMFE(:,frame) = log(abs(R_pos_range));                  
    end
    
% Perform mvn 
    LogMFE_normalised(:,frame)=(LogMFE(:,frame)-mu_x')./stdev_x';  
                                                                    
                                                                    
% wait for first 'past_frames + future_frames+1' (in this case wait for first 5 frames)
% and concatenate input NB features
    if frame < past_frames+1 % for frame = 1 and 2
        perform_ABE=0;  % Do not perform ABE 
        tmp=[tmp; LogMFE_normalised(:,frame)];
        
        % Copy NB frame to extended_speech
        outindex = ((frame-1)*shift_wb+1:(frame-1)*shift_wb+Nfft);
        extended_speech(outindex) = extended_speech(outindex) + [frameNBup(:,frame); zeros(Nfft-wlen_wb,1)]'; 
        
    elseif frame >= past_frames+1 & frame < past_frames+future_frames+1  % for frame =3,4
        perform_ABE = 0; % Do not perform ABE
        tmp=[tmp; LogMFE_normalised(:,frame)];
        
        % Copy NB frame to extended_speech
        outindex = ((frame-1)*shift_wb+1:(frame-1)*shift_wb+Nfft);
        extended_speech(outindex) = extended_speech(outindex) + [frameNBup(:,frame); zeros(Nfft-wlen_wb,1)]'; 
        
    elseif frame==past_frames+future_frames+1 % for frame=5
        perform_ABE=1; % Enable ABE flag to perform ABE for (past_frame+1)the frame (in this case for 3rd frame)
        X_conc(:,j1_ind)=[tmp; LogMFE_normalised(:,frame)];
        
        buffer(:,j1_ind) = X_conc(dimX+1:end,j1_ind); % buffer for 4 frames 
        j1_ind = j1_ind+1;

% Perform ABE for 3rd frame once the 5th frame is received       
    else
        perform_ABE=1;
        X_conc(:,j1_ind)=[buffer(1:end,j1_ind-1); LogMFE_normalised(:,frame)];       
        buffer(:,j1_ind)=X_conc(dimX+1:end,j1_ind);
        j1_ind = j1_ind+1;
    end

if perform_ABE==1 % Perform ABE
    
    inp= X_conc(:,j2_ind) ;

    if strcmp(dr_params.arch,'PCA')
        inp = coeff'*inp;   
        if dr_params.zs==1 % apply mvn on SSAE features
            inp = (inp-mu_x_pca')./(stdev_x_pca');
        end
        
    elseif strfind(dr_params.arch,'SSAE')
% Extract output of middle layer of SSAE
        [~, act] = read_NN(inp', dr_params.NN_weights_biases, dr_params.BN, dr_params.act_list); % inp should be samples x dim
        inp = act{3};
        % Make it a column vector if a row
        if size(inp,1)==1
            inp = inp';
        end
        if dr_params.zs==1 % apply mvn on SSAE features
            inp = (inp-mu_x_AE')./(stdev_x_AE');
        end
    elseif strcmp(dr_params.arch,'CVAE') ||  strcmp(dr_params.arch,'VAE')
        if strcmp(dr_params.arch,'CVAE')
            mu = dr_params.encX_mean.predict(inp');
            log_var = dr_params.encX_var.predict(inp');
        elseif strcmp(dr_params.arch,'VAE')
            mu = dr_params.encX_mean_init.predict(inp');
            log_var = dr_params.encX_var_init.predict(inp');
        end
            
        rng(1); eps_x = randn(size(mu));
        inp = mu + exp(log_var/2) .* eps_x;  
        
        % Make it a column vector if a row
        if size(inp,1)==1
            inp = inp';
        end
        if dr_params.zs==1 % apply mvn on SSAE features
            inp = (inp-mu_x_cvae')./(stdev_x_cvae');
        end
        
    elseif strcmp(dr_params.arch,'SAE')
        inp =  dr_params.encX.predict(inp');
        
        % Make it a column vector if a row
        if size(inp,1)==1
            inp = inp';
        end
        if dr_params.zs==1 % apply mvn on SSAE features
            inp = (inp-mu_x_sae')./(stdev_x_sae');
        end
    end
    
% Perform regression    
    Y_normalised(:,j2_ind) = GMMR(inp, gmmr);

% Perform inverse mvn    
    Y(:,j2_ind) = Y_normalised(:,j2_ind).*stdev_y' + mu_y';
 
% Get LP coefficients, a_eb, for HB and LP gain g_eb 
    a_hb = [1; Y(2:end,j2_ind)];
    g_hb = Y(1,j2_ind);
    
%% Oracle1 - Uncomment following subsection to perform ABE using envlope parameters obtained from original WB frame
%     if fig == 1
%         get_true_eb_env();
% % true env  
%         a_hb=abc;
% % true gain
%         g_hb=sqrt(def);
%     end
    
%% Get HB power spectrum from the estimated LP coefficients and gain 
%  Combine it with NB spectrum followed by Levinson Durbin to get final extended WB spectral envelope
   get_eb_then_combine_is18();
   
%% Extend residual/excitation signal    
    get_res_extend();

%%  Oracle2 - Get original residual
%     get_orig_res();
        
%%
% Subsections Oracle 1 and 2 if uncommented, gives extension using parameters obtained from original HB frame
% In this case, the extension should be exactly same as the original WB signal
% Can be used to confrim the implementation of code 

%%  Uncomment following line to perform ABE using excitation obtained from original WB frame
%     if fig==1
%       get_orig_res();
%     end

%% Get extended signal    
    get_extended_signal();

    if fig==1  
        if j2_ind == plot_frame           
            [a_NBup,e_NBup]=lpc(frameNBup(:,plot_frame),lp_order_wb);  % true NB parameters   
            
            [H_NBup]=freqz(sqrt(e_NBup),a_NBup,Nfft,'whole',Fs16);  
            [H_WB]=freqz(sqrt(e_WB(:,plot_frame)),a_WB,Nfft,'whole',Fs16);  
           plot_fig1();
        end 
    end
    j2_ind = j2_ind+1;
end

% waitbar(frame/Nframes,msg);  

end
% close(msg)
extended_speech(end-Nfft+1:end)=[];  %% combine NB and HB

if fig==1
    plot_fig2()
end
