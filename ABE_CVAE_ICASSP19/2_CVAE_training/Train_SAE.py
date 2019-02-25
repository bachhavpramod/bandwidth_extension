# Written by Pramod Bachhav, August 2018
# Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com

# Code https://github.com/bachhavpramod/bandwidth_extension/blob/master/ABE_SSAE_IS18/2_SSAE_training/SSAE.py
# is modified for single layer output to train a conventional SAE architecture (without pretraining) for dimensionality reduction

##############################################################################################

import keras	
import numpy as np
import os
os.sys.path.append('./../../ABE_SSAE_IS18/2_SSAE_training') # to include files HTK.p, HTKFeat.py and my_functions.py
import my_functions

l1 = 2
l2 = 2
modelpath='./your_models_SAE/' 

if not os.path.exists(modelpath):
    os.makedirs(modelpath) 
    
feature='LPS'
#feature = 'LogMFE'    
print('Feature used is {}'.format(feature))

print( 'Loading data...')
data = my_functions.load_data(l1,l2, feature)        
inp_train, inp_dev, inp_test, op_reg_train, op_reg_dev, op_reg_test, feat_dim_X, feat_dim_Y = data
print('Data loaded') 

###############################################

# Configurations
pDrop = 0; BN = 'b'
HL = [512,256,10,256,512]
activations = ['tanh','tanh','tanh','tanh','tanh','linear']; act='tanh'

################# Training parameters #################

ep = 50  # Number of epochs
ep = 1
optimizer = 'adam'; LR=0.001 # learning rate  
patience = 5; reduce_lr_factor = 0.5; min_LR = 0.00001  # Parameters for callback ReduceLROnPlateau
bs = 512 # batch_size
shuff = True

initializer = keras.initializers.he_normal(seed=7); init='he_n'
loss='mse'

##################################

sheet_name=''
for i in range(len(HL)):
    sheet_name = sheet_name+str(HL[i])
    if i is not len(HL)-1:
        sheet_name=sheet_name+'_'      

arch = 'SAE'
expName = str(len(HL)+1)+'L_'+sheet_name+'_'+arch+'_'+feature+'_NB_LPC_'+str((l1+l2+1)*feat_dim_X)+'.'+feat_dim_Y+'_mem_'+str(l1)+'.'+str(l2)+'_act_'+act+'_dr='+str(pDrop)+'_BN='+str(BN)


L = np.append(feat_dim_X*(l1+l2+1),HL)
L = np.append(L,feat_dim_X*(l1+l2+1))


alpha=0.5
model_name = expName+'_'+optimizer+'_LR='+str(LR)+'_'+loss+'_ep='+str(ep)+'_pat='+str(patience)+'_bs='+str(bs)+init
path_to_save = modelpath+model_name
print('Experiment setup is : '+path_to_save)


from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint

################# Build SSAE architecture #################
 
encoded_layer_index= int((len(L)-1)/2)
num_layers =int((len(L)-1))

inp = Input(shape=(feat_dim_X*(l1+l2+1),))
encoded= inp

for i in np.arange(0, encoded_layer_index):
    if pDrop:
        encoded=Dropout(pDrop)(encoded)
    encoded = Dense(HL[i], kernel_initializer=initializer)(encoded)
    if BN=='b':
        encoded = BatchNormalization()(encoded)
    encoded = Activation(activations[i])(encoded)
    if BN=='a':
        encoded = BatchNormalization()(encoded)
    decoded=encoded

inp_decoder = encoded    
for i in range(encoded_layer_index, num_layers-1):
#    print('i= {}'.format(i))
    if pDrop:
        decoded=Dropout(pDrop)(decoded)
    decoded = Dense(HL[i],  kernel_initializer=initializer)(decoded)
    if BN=='b':
        decoded = BatchNormalization()(decoded)
    decoded = Activation(activations[i])(decoded)
    if BN=='a':
        decoded = BatchNormalization()(decoded)    
    inp_decoder = decoded    

AE_op = Dense( feat_dim_X*(l1+l2+1) , activation=activations[num_layers-1], name='AE')(decoded)
model = Model(inputs = inp, outputs = AE_op)
encoder = Model(inputs = inp, outputs =encoded)

model.compile(optimizer = optimizer, loss='mse')
model.summary()

checkpointer = ModelCheckpoint(filepath=path_to_save+'.hdf5', 
                               verbose=0, save_best_only=True, 
                               monitor='val_loss')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, verbose=1,
                                              patience=patience, min_lr=min_LR)

################# Train SSAE architecture #################

model.fit(inp_train, inp_train,
                epochs = ep,
                batch_size = bs,
                shuffle = shuff,
                validation_data = (inp_dev, inp_dev),
                callbacks = [reduce_lr, checkpointer], 
                verbose = 2,
                )

best_model = keras.models.load_model(path_to_save + '.hdf5')
model_enc = Model(inputs = best_model.inputs, outputs = best_model.get_layer('activation_3').output)
model_enc.save(path_to_save + '_enc.hdf5')

print('----------- Training finished ----------')  
 

