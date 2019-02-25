# Written by Pramod Bachhav, Aug 2018
# Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

# This script loads already trained CVAE models and calculates the RE and KL-d terms
# for a given value of alpha, during training and testing phases:

#    For more details, refer TABLE-1 in the following publication :
# 
#    P.Bachhav, M. Todisco and N. Evans, "Latent Representation Learning for Artificial 
#    Bandwidth Extension using a Conditional Variational Auto-encoder",
#    accepted in ICASSP 2019.

##############################################################################################

import numpy as np
np_seed = 1337
import os
os.environ['PYTHONHASHSEED'] = '0'
import random as rn
rn.seed(12345)
os.environ['KERAS_BACKEND'] = 'theano'
# Loading models with tensorflow gives some error. The quick solution is to load with theano backend
os.sys.path.append('./../../ABE_SSAE_IS18/2_SSAE_training') # to include files HTK.p, HTKFeat.py and my_functions.py
import keras	
import my_functions


l1 = 2 
l2 = 2    
feature='LPS'
print('Feature used is {}'.format(feature))
zy_dim = 10 # size of latent variable of CVAE (zy)
zx_dim = 10 

print( 'Loading data...')
data = my_functions.load_data(l1,l2, feature)        
x_train, x_dev, x_test, y_train, y_dev, y_test, feat_dim_X, feat_dim_Y = data
print('Data loaded') 

#########################################

# Quick check

#l = 3000
#x_train = np.ones((l,1000))
#x_dev = np.ones((l-500,1000))
#x_test = np.ones((l-1000,1000))
#
#y_train = np.ones((l,10))
#y_dev = np.ones((l-500,10))
#y_test = np.ones((l-1000,10))

#########################################    
     
import keras.backend as K
def g_loss(y_true, y_pred):
    return K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)

###############################################

def sampling(z_mean, z_log_var, latent_dim):
    np.random.seed(np_seed) # for reproducibility
    epsilon = np.random.normal( 0., 1.0, (z_mean.shape[0], latent_dim))
    return z_mean + np.exp( z_log_var / 2) * epsilon 

def rec_loss(x, x_pred):
    return (K.sum(K.square(x - x_pred), axis=-1))/alpha

def kl_loss(zy_mean, zy_log_var):
    return - 0.5 * K.sum(1 + zy_log_var - K.square(zy_mean) - K.exp(zy_log_var), axis=-1)

alpha = 5
modelpath='./models_CVAE/'
cvae_file = 'ENC_512_256_DEC_256_512_CVAE_LPS_NB_LPC_200.10_mem_2.2_act_tanh_dr=0_BN=b_adam_LR=0.001_bs=512_ep=50_5_he_n_alpha='+str(alpha)
path_to_save = modelpath + cvae_file

# Load encoder (conditioning network of a CVAE) model trained for a given value of alpha 
model_encX_mean = keras.models.load_model(path_to_save + '_encX_mean.hdf5')
model_encX_var = keras.models.load_model(path_to_save + '_encX_var.hdf5')
model_decY = keras.models.load_model(path_to_save + '_decY.hdf5')
model_encY_mean = keras.models.load_model(path_to_save + '_encY_mean.hdf5')
model_encY_var = keras.models.load_model(path_to_save + '_encY_var.hdf5')

###############################################

print('Get RE and KL-D values during training and testing phase for cvae configuration file : ')
print(cvae_file)

for i in [1,2,3]:

    if i is 1:
        x = x_train; y = y_train
        temp = x_train.shape[0]
        print(" On TRAINING data : ")
    if i is 2: 
        x = x_dev; y = y_dev
        temp = x_dev.shape[0]
        print(" On VALIDATION data: ")
    if i is 3:
        x = x_test; y = y_test
        temp = x_test.shape[0]
        print(" On TEST data ")
    
    # Get latent variable zx
    zx_means = model_encX_mean.predict(x) 
    zx_log_vars= model_encX_var.predict(x) 
    zx = sampling(zx_means, zx_log_vars, zx_dim)
    
    ############## Predict y during TRAINING PHASE #########################
    
    # Sample zy from approximate posterior modeled using 'encY'
    zy_means = model_encY_mean.predict(y) 
    zy_vars = model_encY_var.predict(y) 
    zy_training_phase = sampling(zy_means, zy_vars, 10)
    y_pred_training_phase = model_decY.predict( np.concatenate((zy_training_phase, zx), axis=-1) )  
    
    # Caculate reconstruction error between predicted y and true y
    RE_testing_phase = np.mean(np.sum(np.square(y - y_pred_training_phase), axis=-1) )
    
    ############## Predict y during TESTING PHASE #########################
    
    # Sample zy from Normal distibution
    np.random.seed(np_seed + 100); 
    zy_testing_phase = np.random.normal( 0, 1.0, (temp, zy_dim))     
    y_pred_testing_phase = model_decY.predict( np.concatenate((zy_testing_phase, zx), axis=-1) )  
    
    # Caculate MSE and reconstruction error (RE) between predicted y and true y
    RE_training_phase = np.mean(np.sum(np.square(y - y_pred_testing_phase), axis=-1) )
    
    ############## Calculate KL-D term #########################
    
    # Calculate KD-D term between the approximate posterior and the prior distribution (i.e. Normal)
    KLD = np.mean(-0.5*np.sum(1 + zy_vars - np.square(zy_means) - np.exp(zy_vars), axis=-1)) # first calculate KLD for each sample, then take mean over all samples
    
    # print (file+" === {0:.4f},{1:.4f}".format(score_pred_model[0],score_pred_model[1]))
    print (" Rec error Training phase === {0:.3f}".format(RE_training_phase))
    print (" Rec error Testing phase === {0:.3f}".format(RE_testing_phase))
    print (" KL-D === {0:.10f}".format(KLD))

