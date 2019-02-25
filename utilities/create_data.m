X_train = data_train(:,1:dimX);

%% Apply mean-variance normalisation
stdev_x = std(X_train);  
mu_x = mean(X_train);
X_train_zs = bsxfun(@minus,X_train,mu_x);
X_train_zs = bsxfun(@rdivide,X_train_zs,stdev_x);

Y_train = data_train(:,dimX+1:end);
stdev_y = std(Y_train);  
mu_y = mean(Y_train);
Y_train_zs = bsxfun(@minus,Y_train,mu_y);
Y_train_zs = bsxfun(@rdivide,Y_train_zs,stdev_y);

X_test = data_test(:,1:dimX); X_test_zs=bsxfun(@minus,X_test,mu_x);X_test_zs=bsxfun(@rdivide,X_test_zs,stdev_x);
Y_test = data_test(:,dimX+1:end); Y_test_zs=bsxfun(@minus,Y_test,mu_y);Y_test_zs=bsxfun(@rdivide,Y_test_zs,stdev_y);

X_dev = data_dev(:,1:dimX); X_dev_zs=bsxfun(@minus,X_dev,mu_x);X_dev_zs=bsxfun(@rdivide,X_dev_zs,stdev_x);
Y_dev = data_dev(:,dimX+1:end); Y_dev_zs=bsxfun(@minus,Y_dev,mu_y);Y_dev_zs=bsxfun(@rdivide,Y_dev_zs,stdev_y);

%% Include memory for X
X_train_mem = memory_inclusion2(X_train,l1,l2);
X_train_zs_mem = memory_inclusion2(X_train_zs,l1,l2);
X_test_zs_mem = memory_inclusion2(X_test_zs,l1,l2);
X_dev_zs_mem = memory_inclusion2(X_dev_zs,l1,l2);

%%
X_train = X_train(l1+1:end-l2,:);   X_train_zs = X_train_zs(l1+1:end-l2,:);   
X_test = X_test(l1+1:end-l2,:);  X_test_zs = X_test_zs(l1+1:end-l2,:); 
X_dev = X_dev(l1+1:end-l2,:);  X_dev_zs = X_dev_zs(l1+1:end-l2,:); 

Y_train = Y_train(l1+1:end-l2,:);   Y_train_zs = Y_train_zs(l1+1:end-l2,:);   
Y_test = Y_test(l1+1:end-l2,:);   Y_test_zs = Y_test_zs(l1+1:end-l2,:);  
Y_dev = Y_dev(l1+1:end-l2,:);   Y_dev_zs = Y_dev_zs(l1+1:end-l2,:);  

%%
mu=[mu_x mu_y];  stdev=[stdev_x stdev_y]; 