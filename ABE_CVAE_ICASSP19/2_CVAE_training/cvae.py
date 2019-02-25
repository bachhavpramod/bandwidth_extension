import numpy as np
np_seed = 1337
np.random.seed(np_seed) 
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Lambda, Input
from keras.layers.normalization import BatchNormalization

import keras.backend as K
def g_loss(y_true, y_pred):
    return K.mean(K.square(y_pred[:,0] - y_true[:,0]), axis=-1)

class CVAE(object):

    def __init__(self, parameters):
        self.zx_dim = parameters['zx_dim']
        self.zy_dim = parameters['zy_dim']
        self.optimizer = parameters['optimizer']
        self.optimizer = parameters['optimizer'] 
        self.loss = parameters['loss']
        self.kernel_initializer = parameters['kernel_initializer']
        self.kernel_regularizer = parameters['kernel_regularizer']
        self.epochs = parameters['epochs']
        self.batch_size = parameters['batch_size']
        self.shuff = parameters['shuff']
        self.pi= 3.14
        self.alpha = parameters['alpha']
        
    def init_feedforward(self, L, activations, pDrop, BN, last_layer_name):
#    'last_layer_name' - name for last layer is added. It make reading encoder, decoder parts
#        of VAE easier
        
        self.L = L
        self.BN = BN
        self.pDrop = pDrop
        self.activations =activations
        
        model = Sequential() 
        for i, (n_in, n_out) in enumerate(zip(self.L[:-1], self.L[1:]), start=1):
            print('Training the layer {}: Input {} -> Output {} with activation {}'.format(i, n_in, n_out, self.activations[i-1]))
               
            if i==1:
                if self.pDrop:
                    model.add(Dropout(self.pDrop, input_shape=(n_in,)))
                    model.add(Dense(n_out, kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer))
                if not self.pDrop:
                    model.add(Dense(n_out, kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer,input_shape=(n_in,)))               
                if self.BN=='b':
                    model.add(BatchNormalization())
                model.add(Activation(self.activations[0]))
                if self.BN=='a':
                    model.add(BatchNormalization())
            else:
                if self.pDrop:
                    if i!=len(self.L)-1:
                        model.add(Dropout(self.pDrop))
                model.add(Dense(n_out, kernel_initializer= self.kernel_initializer,
                                kernel_regularizer = self.kernel_regularizer))
                # For only BN layer - apply before each layer untill last    
#                if self.BN=='b':
#                    model.add(BatchNormalization())
#                
#                if i is len(L)-1:     
#                    model.add(Activation(self.activations[i-1], name = last_layer_name))              
#                if self.BN=='a':
#                    model.add(BatchNormalization())
#                    if i is len(L)-1:     
#                        model.add(BatchNormalization(name = last_layer_name))

                if self.BN=='b':
                    model.add(BatchNormalization())              
                if i is len(L)-1:     
                    model.add(Activation(self.activations[i-1], name = last_layer_name)) 
                else:
                    model.add(Activation(self.activations[i-1]))              
                    
                if self.BN=='a':
                    if i is not len(L)-1:     
                        model.add(BatchNormalization())                
        model.summary()
        return model

    def init(self, init_enc_ff_Y, init_dec_ff_Y, encX_mean, encX_var, x_dim, y_dim):

        zy_dim = self.zy_dim
        zx_dim = self.zx_dim
        
        y = Input(shape=(y_dim,), name='op')
        x = Input(shape=(x_dim,), name='inp')
        
        params = {'zy_dim':zy_dim}
        setattr(K, 'params', params)
        # Credits : https://github.com/keras-team/keras/issues/1879
        def sampling(args):
            latent_dim = K.params['zy_dim']
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                      stddev = 1.0)
            return z_mean + ( K.exp(z_log_var/2) ) * epsilon

        zx_mean = encX_mean(x)
        zx_var = encX_var(x)
        zx = Lambda(sampling, output_shape=(zx_dim,), name = 'CVAE_zx')([zx_mean, zx_var])
                     
        h = init_enc_ff_Y(y)
        zy_mean = Dense(zy_dim, name = 'CVAE_z_mean', kernel_initializer= self.kernel_initializer, kernel_regularizer = self.kernel_regularizer)(h)
        zy_log_var = Dense(zy_dim, name = 'CVAE_z_var', kernel_initializer= self.kernel_initializer, kernel_regularizer = self.kernel_regularizer)(h)


        zy = Lambda(sampling, output_shape=(zy_dim,), name = 'CVAE_lambda_z')([zy_mean, zy_log_var])        
        zy_cond = keras.layers.concatenate([zy, zx], axis= -1)         
        op_pred = init_dec_ff_Y(zy_cond)

# Instantiate CVAE model
        self.model = Model([y,x], op_pred)
         
        def rec_loss(x, x_pred):
            return (K.sum(K.square(x - x_pred), axis=-1))/self.alpha
        def kl_loss(x,x_pred):
            return - 0.5 * K.sum(1 + zy_log_var - K.square(zy_mean) - K.exp(zy_log_var), axis=-1)
        def vae_loss(x, x_pred):
            return (rec_loss(x, x_pred) + kl_loss(x,x_pred))
        
        self.model.compile(optimizer=self.optimizer, loss= vae_loss, metrics= [kl_loss, 'mse', rec_loss, g_loss])
        self.model.summary()
              
# Get encoder and decoder models of CVAE       
        self.encY = Model(y, zy)
        self.decY = init_dec_ff_Y    
        
# Get conditioning encoder (mean and variance) model         
        self.encX_mean = Model(x, zx_mean)
        self.encX_var = Model(x, zx_var)
        self.encX = Model(x, zx)
        
    def train(self, data, cb): 
        
        y_train, y_dev, x_train, x_dev = data 
            
        self.model.fit( [y_train, x_train], y_train, epochs=self.epochs,
                        batch_size = self.batch_size,
                        shuffle = self.shuff,
                        validation_data = ([y_dev, x_dev], y_dev),
                        callbacks = cb, 
                        verbose = 2)
        
        model_arch = self.model.to_json()

        encY_arch = self.encY.to_json()
        decY_arch = self.decY.to_json()

        encX_mean_arch = self.encX_mean.to_json()
        encX_var_arch = self.encX_var.to_json()
        encX_arch = self.encX.to_json()
        
        return self.encY, self.decY, self.encX_mean, self.encX_var, self.encX, self.model, model_arch, encY_arch, decY_arch, encX_mean_arch, encX_var_arch, encX_arch

class VAE(object):

    def __init__(self, parameters):
        self.zx_dim = parameters['zx_dim']
        self.optimizer = parameters['optimizer']
        self.optimizer = parameters['optimizer'] 
        self.loss = parameters['loss']
        self.kernel_initializer = parameters['kernel_initializer']
        self.kernel_regularizer = parameters['kernel_regularizer']
        self.epochs = parameters['epochs']
        self.batch_size = parameters['batch_size']
        self.shuff = parameters['shuff']
        self.pi= 3.14
        self.alpha = parameters['alpha']
        
    def init_feedforward(self, L, activations, pDrop, BN, last_layer_name):
        self.L = L
        self.BN = BN
        self.pDrop = pDrop
        self.activations =activations
        
        model = Sequential() 
        for i, (n_in, n_out) in enumerate(zip(self.L[:-1], self.L[1:]), start=1):
            print('Training the layer {}: Input {} -> Output {} with activation {}'.format(i, n_in, n_out, self.activations[i-1]))
               
            if i==1:
                if self.pDrop:
                    model.add(Dropout(self.pDrop, input_shape=(n_in,)))
                    model.add(Dense(n_out, kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer))
                if not self.pDrop:
                    model.add(Dense(n_out, kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=self.kernel_regularizer,input_shape=(n_in,)))               
                if self.BN=='b':
                    model.add(BatchNormalization())
                model.add(Activation(self.activations[0]))
                if self.BN=='a':
                    model.add(BatchNormalization())
            else:
                if self.pDrop:
                    if i!=len(self.L)-1:
                        model.add(Dropout(self.pDrop))
                model.add(Dense(n_out, kernel_initializer= self.kernel_initializer,
                                kernel_regularizer = self.kernel_regularizer))
#                if self.BN=='b':
#                    model.add(BatchNormalization())
#                model.add(Activation(self.activations[i-1]))              
#                if self.BN=='a':
#                    model.add(BatchNormalization())                   
#                if i is len(L)-1:     
#                    model.add(Activation(self.activations[i-1],  name = last_layer_name))              
#                if self.BN=='a':
#                    model.add(BatchNormalization())
#                    if i is len(L)-1:     
#                        model.add(BatchNormalization(name = last_layer_name))                    
                if self.BN=='b':
                    model.add(BatchNormalization())              
                if i is len(L)-1:     
                    model.add(Activation(self.activations[i-1], name = last_layer_name)) 
                else:
                    model.add(Activation(self.activations[i-1]))              
                    
                if self.BN=='a':
                    if i is not len(L)-1:     
                        model.add(BatchNormalization())
        model.summary()
        return model

    def init(self, init_encoder_ff_X, init_decoder_ff_X, input_dim):

        zx_dim = self.zx_dim
        
        x = Input(shape=(input_dim,))
        h = init_encoder_ff_X(x)
        z_mean = Dense(zx_dim, name ='VAE_z_mean',  kernel_initializer= self.kernel_initializer, kernel_regularizer = self.kernel_regularizer)(h)
        z_log_var = Dense(zx_dim, name ='VAE_z_var', kernel_initializer= self.kernel_initializer, kernel_regularizer = self.kernel_regularizer)(h)
        
        params = {'latent_dim1':zx_dim}
        setattr(K, 'params', params)
        # Credits : https://github.com/keras-team/keras/issues/1879
        def sampling1(args):
            latent_dim = K.params['latent_dim1']
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                      stddev = 1.0)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        z = Lambda(sampling1, output_shape=(zx_dim,), name = 'VAE_lambda_z')([z_mean, z_log_var])
        x_pred = init_decoder_ff_X(z)

# Instantiate VAE model
        self.model = Model(x, x_pred)

        def rec_loss(x, x_pred):
            return (K.sum(K.square(x - x_pred), axis=-1))/self.alpha
        def kl_loss(x,x_pred):
            return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        def vae_loss(x, x_pred):
            return (rec_loss(x, x_pred) + kl_loss(x,x_pred)) ## keras examples master
        
        self.model.compile(optimizer=self.optimizer, loss= vae_loss, metrics= [kl_loss, 'mse', rec_loss])
        self.model.summary()
                
        self.enc_check = init_encoder_ff_X
        self.enc = Model(x, z)
        
        self.enc_mean = Model(x, z_mean)
        self.enc_var = Model(x, z_log_var)
        self.dec = init_decoder_ff_X
        
    def train(self, data, cb):

        inp_train, inp_dev, op_train, op_dev = data
        self.model.fit(inp_train, op_train, epochs=self.epochs,
                        batch_size = self.batch_size,
                        shuffle = self.shuff,
                        validation_data = (inp_dev, op_dev),
                        callbacks = cb, 
                        verbose = 2,)
            
        model_arch = self.model.to_json()
        enc_arch = self.enc.to_json()
        dec_arch = self.dec.to_json()
                
        return self.enc, self.enc_mean, self.enc_var, self.dec, self.model, model_arch, enc_arch, dec_arch, self.enc_check
   
