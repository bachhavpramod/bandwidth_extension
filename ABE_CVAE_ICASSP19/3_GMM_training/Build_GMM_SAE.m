clc; close all;
addpath('./../../utilities')

%% Get data

inp_feature = 'PS'; dimX = 200; dimY = 10;

filename = [inp_feature,'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=',num2str(dimX),'_HB=10'];
disp('Loading data')

path='./../1_Feature_extraction/';

data1 = readhtk([path,filename,'_train1']); % one frame per row
data2 = readhtk([path,filename,'_train2']); % one frame per row
data_train = [data1; data2];

data_dev = readhtk([path,filename,'_dev']); % one frame per row
data_test = readhtk([path,'TSP_',filename]); % one frame per row
    
data_train = data_dev;
data_test = data_dev;

clear data1 data2 data3

l1 = 2; l2 = 2;
if strcmp(inp_feature, 'PS')
    data_train(:,1:dimX)= log(abs(data_train(:,1:dimX)));
    data_dev(:,1:dimX)= log(abs(data_dev(:,1:dimX)));
    data_test(:,1:dimX)= log(abs(data_test(:,1:dimX)));
    inp_feature = 'LPS';
end
create_data();
disp('Data loaded')

disp(['Train data :', num2str(size(data_train))])
disp(['Dev data :', num2str(size(data_dev))])
disp(['Test data :', num2str(size(data_test))])    

%% 
path_to_SAE_models='./../2_CVAE_training/models_SAE/';
file = '6L_512_256_10_256_512_SAE_LPS_NB_LPC_1000.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_mse_ep=50_pat=5_bs=512_he_n';
encX = importKerasNetwork([path_to_SAE_models,file,'_enc.hdf5'],'OutputLayerType','regression')

%%
disp(['SAE configuration = ', file])

%%

inp_train = reshape(X_train_zs_mem',[1,dimX*(l1+l2+1),1,size(X_train_zs_mem,1)]); 
inp_dev = reshape(X_dev_zs_mem',[1,dimX*(l1+l2+1),1,size(X_dev_zs_mem,1)]);
inp_test = reshape(X_test_zs_mem',[1,dimX*(l1+l2+1),1,size(X_test_zs_mem,1)]); 

bottleneck_train = encX.predict(inp_train);
bottleneck_dev = encX.predict(inp_dev);
bottleneck_test = encX.predict(inp_test);

%% 

[bottleneck_train_zs, mu_x_sae, stdev_x_sae] = mvn_train(bottleneck_train);
bottleneck_dev_zs = mvn_test(bottleneck_dev, mu_x_sae, stdev_x_sae);
bottleneck_test_zs = mvn_test(bottleneck_test, mu_x_sae, stdev_x_sae);

Initialization='Random';
disp(['Initialization = ',Initialization])

N = size(Y_train_zs,1); % number of frames
comp = 128; iter = 100; 
disp(['Number of frames = ',num2str(N)])
disp(['Components = ',num2str(comp)])
fprintf('\n')
disp('*******************************************************')    
fprintf('\n')

%%
for j=0:1
    
if j==0
    Z=[bottleneck_train Y_train_zs]; 
else
    Z=[bottleneck_train_zs Y_train_zs];
end

    disp(['Building GMM ',num2str(j)]) 
    disp(['size of Z = ',num2str(size(Z))])
    fprintf('\n')
    disp(['First and last elements of first vector of Z are ',num2str(Z(1,1)),' and ',num2str(Z(end,1))])        
    tic
    OPTIONS = statset('MaxIter',iter,'Display','iter','TolFun',1e-4);
    rng(1)
    obj = gmdistribution.fit(Z,comp,'CovType','full','Regularize',1e-1,'Options',OPTIONS);  %% Z for gmdistribution.fit is FRAME X DIM
    disp('finished')
    time = toc;
    disp(['Time taken : ',num2str(time/60), ' minutes'])

fprintf('\n')
disp('*******************************************************')    

if j==1
    GMMfile_zs = [file,'_zs=1','_GMM_',num2str(comp),'.mat'];
    save(GMMfile_zs, 'obj','mu','stdev','mu_x_sae','stdev_x_sae')

else
    GMMfile=[file,'_zs=0','_GMM_',num2str(comp),'.mat'];
    save(GMMfile, 'obj','mu','stdev')
end
end

%%
disp('Calculating MSE')
disp(['SAE configuration ',  file])
dim = size(Y_test, 2);

%%
for i=1:2
    
if i==1    
    disp('for zs=0')
    disp(['Loading GMM model ... = ', GMMfile])
    load(GMMfile)
    
    feat_test_sae = bottleneck_test;
    feat_dev_sae = bottleneck_dev;
    
elseif i==2
   
    disp('for zs=1')
    disp(['Loading GMM model ... = ', GMMfile_zs])
    load(GMMfile_zs)
     
    feat_test_sae = bottleneck_test_zs;
    feat_dev_sae = bottleneck_dev_zs;    
end

    ComponentMean = obj.mu';   % means for every component (every model) % (dimX+dimY) x NumOfComp
    ComponentVariance = obj.Sigma;
    apriory_prob = obj.PComponents;
    gmmr = offline_param(apriory_prob,ComponentMean,ComponentVariance,dim);

disp('MSE calculation on DEV set')
    Y_dev_est_zs=[];
    for m = 0:size(feat_dev_sae,1)-1
        Y_dev_est_zs(m+1,:) = GMMR(feat_dev_sae(m+1,:)', gmmr)';   
    end

    Y_dev_est = bsxfun(@times, Y_dev_est_zs, stdev_y); Y_dev_est = bsxfun(@plus, Y_dev_est, mu_y);
%     Y_dev_est = inverse_mvn(Y_dev_est_zs)  bsxfun(@times, Y_dev_est_zs, stdev_y); Y_dev_est = bsxfun(@plus, Y_dev_est, mu_y);
    fprintf('\n')
    disp(['MSE dev = ',num2str(mean(mean((Y_dev_est-Y_dev).^2)))])
    disp(['MSE gain - dev  = ',num2str(mean((Y_dev_est(:,1)-Y_dev(:,1)).^2))])
    disp(['MSE dev - on mvn data = ',num2str(mean(mean((Y_dev_est_zs-Y_dev_zs).^2)))])
    disp(['MSE gain dev - on mvn data = ',num2str(mean((Y_dev_est_zs(:,1)-Y_dev_zs(:,1)).^2))])
    fprintf('\n')   

disp('MSE calculation on TEST set')
    Y_test_est_zs =[];
    for m = 0:size(feat_test_sae,1)-1
        Y_test_est_zs(m+1,:) = GMMR(feat_test_sae(m+1,:)', gmmr)';   
    end

    Y_test_est = bsxfun(@times,Y_test_est_zs,stdev_y); Y_test_est = bsxfun(@plus,Y_test_est,mu_y);
    fprintf('\n')
    disp(['MSE test = ',num2str(mean(mean((Y_test_est-Y_test).^2)))])
    disp(['MSE gain - test  = ',num2str(mean((Y_test_est(:,1)-Y_test(:,1)).^2))])
    disp(['MSE test - on mvn data = ',num2str(mean(mean((Y_test_est_zs-Y_test_zs).^2)))])
    disp(['MSE gain test - on mvn data = ',num2str(mean((Y_test_est_zs(:,1)-Y_test_zs(:,1)).^2))])
    fprintf('\n')   
    disp('*******************************************************')    
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Finished')


