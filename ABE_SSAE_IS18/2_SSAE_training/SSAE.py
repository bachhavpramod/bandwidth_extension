# Written by Pramod Bachhav, June 2018
# Contact : bachhav[at]eurecom[dot]fr, bachhavpramod[at]gmail[dot]com
#
#   References:
# 
#    P.Bachhav, M. Todisco and N. Evans, "Artificial Bandwidth Extension 
#    with Memory Inclusion using Semi-supervised Stacked Auto-encoders", 
#    to appear in Proceedings of INTERSPEECH, Hyderabad, India.
# 
#   Acknowledgements : thanks to the following links and many others 
#    https://keras.io/getting-started/functional-api-guide/
#    https://keras.io/getting-started/functional-api-guide/
#    https://blog.keras.io/building-autoencoders-in-keras.html
#    https://machinelearningmastery.com/check-point-deep-learning-models-keras/

##############################################################################################

import keras	
import my_functions
import numpy as np
import os

l1=2
l2=2
modelpath='./your_models_SSAE/' 

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
# Configurations - for logMFE
#pDrop = 0.2; BN = 0 # Fraction of the input units to drop.
#pDrop=0; BN='a'
pDrop=0; BN='b'
#pDrop=0; BN=0
#HL=[512,256,10,256,512]
HL = [1024,512,10,512,1024]

activations = ['tanh','tanh','tanh','tanh','tanh','linear']; act='tttttl'
#activations=['relu','relu','relu','relu','relu','linear']; act='rrrrrl'

#################
## Configurations - for LPS
#activations=['tanh','tanh','tanh','tanh','tanh','linear']; act='tttttl'
#pDrop=0; BN='b'
##HL=[1024,512,10,512,1024]
#HL=[512,256,10,256,512]

################# Training parameters #################

ep = 1  # number of epochs
optimizer = 'adam'; LR=0.001 # learning rate  
patience = 1; reduce_lr_factor = 0.5; min_LR = 0.00001  # parameters for callback ReduceLROnPlateau
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

arch = 'SSAE'
expName = str(len(HL)+1)+'L_'+sheet_name+'_'+arch+'_'+feature+'_NB_LPC_'+str(feat_dim_X)+'.'+str(feat_dim_Y)+'_mem_'+str(l1)+'.'+str(l2)+'_act_'+act+'_dr='+str(pDrop)+'_BN='+str(BN)


L = np.append(feat_dim_X*(l1+l2+1),HL)
L = np.append(L,feat_dim_X*(l1+l2+1))


alpha=0.5
model_name = expName+'_'+optimizer+'_LR='+str(LR)+'_'+loss+'_ep='+str(ep)+'_bs='+str(bs)+'_shuff='+init+'_alpha='+str(alpha)
path_to_save = modelpath+model_name
print('Experiment setup is : '+path_to_save)


from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
import keras.backend as K
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
reg_op = Dense(feat_dim_Y, activation=activations[num_layers-1], name='reg')(decoded)
model = Model(inputs = inp, outputs = [AE_op,  reg_op])
encoder = Model(inputs = inp, outputs =encoded)

# Loss function to monitor mse only on gain coefficient
def g_loss(y_true, y_pred):
    return K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)


model.compile(optimizer = optimizer, loss={'AE' : 'mse' , 'reg' : 'mse'},  loss_weights=[alpha, 1-alpha], 
              metrics= {'reg': g_loss})
model.summary()

checkpointer = ModelCheckpoint(filepath=path_to_save+'.hdf5', 
                               verbose=0, save_best_only=True, 
                               monitor='val_loss')

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, verbose=1,
                                              patience=patience, min_lr=min_LR)

################# Train SSAE architecture #################

model.fit(inp_train, [inp_train, op_reg_train],
                epochs = ep,
                batch_size = bs,
                shuffle = shuff,
                validation_data = (inp_dev, [inp_dev, op_reg_dev]),
                callbacks = [reduce_lr, checkpointer], 
                verbose = 2,
                )
print('----------- Training finished ----------')  
 
################# Evaluation #################
# NOTE :To load successfully this model, copy g_loss definition to 
# your_environment\lib\site-packages\keras\metrics.py

print('Test evaluation :')    
print("The final Model is in :  ",path_to_save)  
print('Load the model with least val loss')
best_model = keras.models.load_model(path_to_save+'.hdf5')
score_test = best_model.evaluate(inp_test, [inp_test, op_reg_test], batch_size=16, verbose=0)    
print ("Test score - best model using model.evaluate : {0:.3f},{1:.3f},{2:.3f},{3:.3f}".format(score_test[0],score_test[1],score_test[2],score_test[3]))
print('----------- Evaluation finished ----------')   


