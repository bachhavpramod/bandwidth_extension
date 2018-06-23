import numpy as np
from copy import deepcopy
import HTK as htk 
from sklearn.preprocessing import scale

def load_data(l1, l2, feature):
  
    features = htk.HTKFile()
    
    if feature=='LogMFE':
        path_train='./../1_Feature_extraction/'+feature+'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10_train'
        path_dev='./../1_Feature_extraction/'+feature+'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10_dev'
        path_test='./../1_Feature_extraction/'+feature+'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10_test'
        path_TSP='./../1_Feature_extraction/TSP_'+feature+'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=10_HB=10'
        dimX=10
    
    if feature=='PS' or feature=='LPS':
        path_train='./../1_Feature_extraction/PS'+'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=200_HB=10_train'
        path_dev='./../1_Feature_extraction/PS'+'_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=200_HB=10_dev'
        path_test='./../1_Feature_extraction/PS_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=200_HB=10_test'
        path_TSP='./../1_Feature_extraction/TSP_PS_NB_300_3400Hz_LPC_WB_3400_8000_Hz_20_10ms_dim_LB=200_HB=10'
        dimX=200
    

    # Load training data
    features.load(path_train)
    data1= np.asarray(features.data)  # featdim X number of frames
    #Load dev data
    features.load(path_dev)
    data2= np.asarray(features.data)    
    # Load test data
    features.load(path_test)
    data3= np.asarray(features.data)


    data_train= np.concatenate((data1,data2),axis=0)
    data_dev= data3
    features.load(path_TSP)
    data_test= np.asarray(features.data)  # featdim X number of frames
   
    
    X_train=data_train[:,0:dimX]
    Y_train=data_train[:,dimX:] 

    X_dev=data_dev[:,0:dimX]
    Y_dev=data_dev[:,dimX:]

    X_test=data_test[:,0:dimX]
    Y_test=data_test[:,dimX:]
        
    if feature=='LPS':
        print('Log is applied')
        X_train=np.log(np.abs(X_train))
        X_test=np.log(np.abs(X_test))
        X_dev=np.log(np.abs(X_dev))
        
# Apply mean-variance normalisation on training data    
    sX_train = scale(X_train, axis = 0) 
    sY_train = scale(Y_train, axis = 0) 

# Get means and variances from training data
    mean_trainX=X_train.mean(axis=0)
    std_trainX=X_train.std(axis=0)   
    mean_trainY=Y_train.mean(axis=0)
    std_trainY=Y_train.std(axis=0)
     
# Normalize dev and test data
    sX_dev=(X_dev-mean_trainX)/std_trainX
    sX_test=(X_test-mean_trainX)/std_trainX
    sY_dev=(Y_dev-mean_trainY)/std_trainY
    sY_test=(Y_test-mean_trainY)/std_trainY

#  Include memory
    sX_train=memory_inclusion2(sX_train,l1,l2)        
    sX_dev=memory_inclusion2(sX_dev,l1,l2)        
    sX_test=memory_inclusion2(sX_test,l1,l2) 
  
    if (l1!=0  and l2!=0):
        sY_train=sY_train[l1:-l2,:]
        sY_dev=sY_dev[l1:-l2,:]
        sY_test=sY_test[l1:-l2,:]

    feat_dim_X=int(X_train.shape[1])
    feat_dim_Y=int(Y_train.shape[1])

    return [sX_train, sX_dev, sX_test, sY_train, sY_dev, sY_test, feat_dim_X, feat_dim_Y]


def memory_inclusion2(X,l1,l2): 
    temp=0
    if np.size(X,1)<np.size(X,0):
        X=X.T
        temp=1
        
        # X should have all features/observations as columns i.e feat_dim x frames
    dimX= X.shape[0]
    N= X.shape[1]
    X2=np.zeros( (dimX*(l1+l2+1), N) )
        
    for i in range(0,(l1+l2+1)):
        X2 [ dimX*i : dimX*(i+1) , 0:N-i ]= deepcopy (X[ :,i:]);
            
    X2=np.delete(X2,np.arange(N-l2-l1,N),1) # remove first l1 column
    if temp:
        X2=X2.T
    return X2    
