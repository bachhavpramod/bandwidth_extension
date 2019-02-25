% Script to train a Gaussian mixture model (GMM) using 10d NB features vectors
% obtained by applying PCA to 1000d log-spectral coefficients with memory included from 
% 2 neighbouring frames and HB feature vectors

%%

% Use Build_GMM_CVAE.m or Build_GMM_SAE.m for reference
% Only difference is apply PCA as a DR technique to the data with memory 

% coeff = pca(X_train_zs_mem,'Centered',false,'Algorithm','eig','NumComponents',10); 
% X_train_zs_mem_pca = X_train_zs_mem* coeff;
% X_dev_zs_mem_pca = X_dev_zs_mem* coeff;
% X_test_zs_mem_pca = X_test_zs_mem* coeff;
% 
% [X_train_zs_mem_pca_zs, mu_x_pca, stdev_x_pca] = mvn_train(X_train_zs_mem_pca);
% [X_dev_zs_mem_pca_zs] = mvn_test(X_dev_zs_mem_pca, mu_x_pca, stdev_x_pca);
% [X_test_zs_mem_pca_zs] = mvn_test(X_test_zs_mem_pca, mu_x_pca, stdev_x_pca);
% 
% 
% inp_feature = 'LPS_pca';
% filename = [inp_feature,'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10'];
