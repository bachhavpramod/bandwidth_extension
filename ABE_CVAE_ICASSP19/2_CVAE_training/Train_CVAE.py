# A keras script to train a conditional variational auto-encoder (CVAE)
# with a conditioning variable optimised via an auxillary neural network.
# The conditioning network is further used for dimensionality reduction 
# to perform Artificial bandwidth extension using GMM regression. 
#
# Written by Pramod Bachhav, Aug 2018
# Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
#
#   References:
# 
#    P.Bachhav, M. Todisco and N. Evans, "Latent Representation Learning for Artificial 
#    Bandwidth Extension using a Conditional Variational Auto-encoder",
#    accepted in ICASSP 2019.
# 
#   Acknowledgements : 
#
#     https://github.com/twolffpiggott/autoencoders/blob/master/autoencoders.py#L34
#     https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
#
#     https://blog.keras.io/building-autoencoders-in-keras.html
#        - Here, note that KL loss definition has a minor mistake - 
#        - It should be KL-loss = -0.5 K.sum() .... instead of KL-loss = -0.5 K.mean()
#     Better version can be found at => 
#     https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
#
#    A very nice understanding about Kingma's papers on VAE at 
#     http://bjlkeng.github.io/posts/variational-autoencoders/
    
##############################################################################################

import numpy as np
np_seed = 1337
import os
os.environ['PYTHONHASHSEED'] = '0'
os.sys.path.append('./../../ABE_SSAE_IS18/2_SSAE_training') # to include files HTK.p, HTKFeat.py and my_functions.py
import random as rn
rn.seed(12345)
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras	
np.random.seed(np_seed) 
from keras.callbacks import ModelCheckpoint
import my_functions
from keras.models import Model
import cvae


l1 = 2 
l2 = 2
modelpath='./your_models_CVAE/'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)
     
feature='LPS'
print('Feature used is {}'.format(feature))

print( 'Loading data...')
data = my_functions.load_data(l1,l2, feature)        
inp_train, inp_dev, inp_test, op_reg_train, op_reg_dev, op_reg_test, feat_dim_X, feat_dim_Y = data
print('Data loaded') 
feat_dim_X = (l1+l2+1)*feat_dim_X

# =============================================================================
# CVAE configuration
# =============================================================================

zy_dim = 10 # size of latent variable of CVAE (zy)
zx_dim = 10 # size of conditioning variable of CVAE (zx)

alpha_vae = 10
alpha_cvae = 10
  
hidden_layers_enc=[512, 256]; 
hidden_layers_dec=[256, 512]; 

activ = 'tanh'; act = 'tanh'
activations_enc =[activ,activ]; 
activations_dec =[activ,activ,'linear'] 

L_enc_X = np.append(feat_dim_X,hidden_layers_enc) 
L_dec_X = np.append( zx_dim, hidden_layers_dec); L_dec_X = np.append( L_dec_X , feat_dim_X)

L_enc_Y = np.append( feat_dim_Y, hidden_layers_enc); 
L_dec_Y = np.append( zy_dim + zx_dim , hidden_layers_dec); L_dec_Y = np.append( L_dec_Y , feat_dim_Y)

# =============================================================================
# training parameters
# =============================================================================

pDrop = 0; BN = 'b'
reduce_lr_factor = 0.5; min_lr = 0.00001  # parameters for callback ReduceLROnPlateau
bs = 512 # batch_size
shuff = True
loss='mse'

epochs = 50; epochs_cvae = 50; patience = 5; patience_cvae = 5
LR = 0.001; optimizer = 'adam'; 
batch_size = 512
shuff = True; loss = 'mse'

optim1 = keras.optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)  # 0.87, 0.88, 0.90 decay - 0.0, bs=128
optim2 = keras.optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)  # 0.87, 0.88, 0.90 decay - 0.0, bs=128
opt_params = 'LR='+str(LR)

kernel_initializer = keras.initializers.he_normal(seed=7); init='he_n'
kr = 0; kernel_regularizer = keras.regularizers.l2(kr)

################################################################

sheet_name1 = ''
for i in range(len(hidden_layers_enc)):
    sheet_name1 = sheet_name1 + str(hidden_layers_enc[i])
    if i is not len(hidden_layers_enc)-1:
        sheet_name1 = sheet_name1+'_' 
sheet_name1 = 'ENC_' + sheet_name1 
sheet_name2 = ''
for i in range(len(hidden_layers_dec)):
    sheet_name2 = sheet_name2 + str(hidden_layers_dec[i])
    if i is not len(hidden_layers_dec)-1:
        sheet_name2 = sheet_name2+'_'        
sheet_name2 = 'DEC_' + sheet_name2 
        
################################################################
arch = 'CVAE'     
expName=sheet_name1+'_'+sheet_name2+'_'+arch+'_'+feature+'_NB_LPC_'+str(int(feat_dim_X/(l1+l2+1)))+'.'+str(feat_dim_Y)+'_mem_'+str(l1)+'.'+str(l2)+'_act_'+act+'_dr='+str(pDrop)+'_BN='+str(BN)# HL+1-OL
model_name = expName+'_'+optimizer+'_'+opt_params+'_bs='+str(batch_size)+'_ep='+str(epochs_cvae)+'_'+str(patience_cvae)+'_'+init+'_alpha='+str(alpha_cvae)

################################################################

data_vae = inp_train, inp_dev, inp_train, inp_dev
data_cvae = op_reg_train, op_reg_dev, inp_train, inp_dev 

################################################################
if not os.path.exists(modelpath):
    os.makedirs(modelpath)  
path_to_save = modelpath+model_name
print('Experiment setup is : '+path_to_save)
    
def sampling(z_mean, z_log_var, latent_dim):
    np.random.seed(np_seed) # for reproducibility
    epsilon = np.random.normal( 0., 1.0, (z_mean.shape[0], latent_dim))
    return z_mean + np.exp( z_log_var / 2) * epsilon 

print('*******************************************************************')
print('             TRAINING A VAE for X           ')
print('*******************************************************************')

monitor = 'val_loss'
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = monitor, factor=0.5, 
                                              verbose = 1, patience = patience, min_lr = min_lr)
checkpointer = ModelCheckpoint(filepath = path_to_save+'_VAEx_init', verbose = 0, 
                               save_best_only = True, monitor= monitor)
cb  = [reduce_lr, checkpointer]

parameters = {'zx_dim': zx_dim, 'optimizer':optim1, 'loss':loss,
              'epochs' : epochs, 'batch_size' : batch_size, 'shuff' : shuff,
              'kernel_initializer':kernel_initializer, 'kernel_regularizer' :kernel_regularizer,
              'alpha' : alpha_vae}

vae_instance = cvae.VAE (parameters)

print('--------------- Initializing encoder network for X --------------- ')
init_encoder_ff_X = vae_instance.init_feedforward( L_enc_X , activations_enc, pDrop, BN, 'encX_last_layer' )
print('--------------- Initializing decoder network for X ---------------')
init_decoder_ff_X = vae_instance.init_feedforward( L_dec_X, activations_dec, pDrop, BN, 'reconstructed_x')
print('--------------- Initializing VAE for X --------------- ')
w_encX_init = init_encoder_ff_X.get_weights()
w_decX_init = init_decoder_ff_X.get_weights()
vae_instance.init(init_encoder_ff_X, init_decoder_ff_X, feat_dim_X)
print('--------------- Training VAE for X---------------')
encX, encX_mean, encX_var, decX, vaeX, vae_arch, encX_arch, decX_arch, encX_check = vae_instance.train (data_vae, cb) 


# Weights of best model with best validation loss are saven in (path_to_save+'_VAEx_initial') 
# load best weights in model vaeX
w_vaeX = vaeX.get_weights()
vaeX.load_weights(path_to_save+'_VAEx_init')
w_vaeX_best = vaeX.get_weights()

# MATLAB, yet (Oct 2018), does not support loading keras models with lambda layer. 
# Therefore, it is not possible to use the model enc_x
encX =  Model(inputs=vaeX.input, outputs=vaeX.get_layer('VAE_lambda_z').output)
# Save models for mean and variances of stochastic layer 'zx', separately.
encX_mean =  Model(inputs = vaeX.input, outputs=vaeX.get_layer('VAE_z_mean').output)
encX_var =  Model(inputs = vaeX.input, outputs=vaeX.get_layer('VAE_z_var').output)

# this workaround helpful to read these models in MATLAB during ABE estimation, using importkeras add-on. 
encX_mean.save(path_to_save+'_encX_mean_init.hdf5')
encX_var.save(path_to_save+'_encX_var_init.hdf5')


print('*******************************************************************')
print('             TRAINING A CVAE           ')
print('*******************************************************************')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = monitor, factor=0.5, 
                                              verbose = 1, patience = patience_cvae, min_lr = min_lr)
checkpointer = ModelCheckpoint(filepath = path_to_save+'_CVAE', verbose = 0, 
                               save_best_only = True, monitor= monitor)
cb  = [reduce_lr, checkpointer]
parameters['optimizer'] = optim2
parameters['alpha'] = alpha_cvae
parameters['zy_dim'] = zy_dim

cvae_instance = cvae.CVAE (parameters)

print('             Initializing encoder network          ')
init_encoder_ff_Y = cvae_instance.init_feedforward( L_enc_Y , activations_enc, pDrop, BN, 'encY_last_layer' )
print('             Initializing decoder network          ')
init_decoder_ff_Y = cvae_instance.init_feedforward( L_dec_Y, activations_dec, pDrop, BN, 'reconstructed_y' )
w_decY_init = init_decoder_ff_Y.get_weights()
w_encY_init = init_encoder_ff_Y.get_weights()
print('             Initializing CVAE              ')
cvae_instance.init(init_encoder_ff_Y, init_decoder_ff_Y, encX_mean, encX_var, feat_dim_X, feat_dim_Y)
print('             Training CVAE              ')
encY, decY, encX_mean_new, encX_var_new, encX_new, cvae, cvae_arch, encY_arch, decY_arch, encX_mean_arch, encX_var_arch, encX_arch = cvae_instance.train(data_cvae, cb) 
decY = []


# load best weights in 'cvae'
cvae.load_weights(path_to_save+'_CVAE')

#encY_mean_new =  Model(inputs = cvae.input, outputs = cvae.get_layer('CVAE_z_mean').output)
#encY_var_new =  Model(inputs = cvae.input, outputs = cvae.get_layer('CVAE_z_var').output)

# Dec is last 'sequential model of' cvae
decY = cvae.get_layer(index = len(cvae.layers)-1)

# zx_mean and zx_var should be 6th and 7th sequential layers
encX_mean_best = cvae.get_layer(index = 5)
encX_mean_best.summary()
encX_var_best = cvae.get_layer(index = 6)
encX_var_best.summary()
encX_best =  Model(inputs = cvae.input[1], outputs = cvae.get_layer('CVAE_zx').output)
encX_best.summary()

encY_mean_best =  Model(inputs = cvae.input[0], outputs = cvae.get_layer('CVAE_z_mean').output)
encY_mean_best.summary()
encY_mean_best.input

encY_var_best =  Model(inputs = cvae.input[0], outputs = cvae.get_layer('CVAE_z_var').output)
encY_var_best.summary()
encY_var_best.input

# Save models
encX_best.save(path_to_save+'_encX.hdf5')
encX_mean_best.save(path_to_save+'_encX_mean.hdf5')
encX_var_best.save(path_to_save+'_encX_var.hdf5')
decY.save(path_to_save+'_decY.hdf5')


# =============================================================================
# Evaluation of the model (testing phase - where zy is sampled from prior distribution) 
# on train, validation/developement and test  dataset
# =============================================================================

# Sample 'zy' from Normal distribution  during estimation phase
np.random.seed(11); zy_train = np.random.normal( 0, 1.0, (inp_train.shape[0], zy_dim))     
np.random.seed(12); zy_dev = np.random.normal( 0, 1.0, (inp_dev.shape[0], zy_dim))     
np.random.seed(13); zy_test = np.random.normal( 0, 1.0, (inp_test.shape[0], zy_dim))     
 

print('*******************************************************************')
print('             EVALUATION - with best weights for encX and decY  ')
print('*******************************************************************')

means_train = encX_mean_best.predict(inp_train) 
log_vars_train = encX_var_best.predict(inp_train) 
zx_train = sampling(means_train, log_vars_train, zx_dim)

op_reg_train_est0 = decY.predict( np.concatenate((zy_train, zx_train), axis=-1) )  
score_pred_model0 = np.append(np.mean(np.square(op_reg_train - op_reg_train_est0)), np.mean(np.square(op_reg_train[:,0]-op_reg_train_est0[:,0])))
print ("Train score : {0:.3f},,{1:.7f}".format(score_pred_model0[0],score_pred_model0[1]))

# ---------------------------------------------------------------------------

means_dev = encX_mean_new.predict(inp_dev) 
log_vars_dev = encX_var_new.predict(inp_dev) 
zx_dev = sampling(means_dev, log_vars_dev, zx_dim)

op_reg_dev_est1 = decY.predict( np.concatenate((zy_dev, zx_dev), axis=-1) )  
score_pred_model1 = np.append(np.mean(np.square(op_reg_dev - op_reg_dev_est1)), np.mean(np.square(op_reg_dev[:,0]-op_reg_dev_est1[:,0])))
print ("Dev score : {0:.3f},,{1:.7f}".format(score_pred_model1[0],score_pred_model1[1]))

# ---------------------------------------------------------------------------

means_test = encX_mean_new.predict(inp_test) 
log_vars_test = encX_var_new.predict(inp_test) 
zx_test = sampling(means_test, log_vars_test, zx_dim)

op_reg_est2 = decY.predict( np.concatenate((zy_test, zx_test), axis=-1) )  
score_pred_model2 = np.append(np.mean(np.square(op_reg_test - op_reg_est2)), np.mean(np.square(op_reg_test[:,0]-op_reg_est2[:,0])))
print ("Test score : {0:.3f},,{1:.7f}".format(score_pred_model2[0],score_pred_model2[1]))

  
##################################################################
#
print('----------- Evaluation finished ----------')  
print('-----------Experiment setup is : ' +path_to_save + '-----------')

#################################################################


