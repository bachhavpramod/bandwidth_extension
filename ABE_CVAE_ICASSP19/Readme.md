# Description:
These folders contain scripts to perform dimensionality reduction (DR)/ feature extraction using conditional variational auto-encoder (CVAE) and perform artificial bandwidth extension using [GMM regression](https://github.com/bachhavpramod/bandwidth_extension/blob/master/utilities/GMMR.m). 
____________________________________________________________________________
Copyright (C) 2019 EURECOM, France.

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/
____________________________________________________________________________

For further details refer to following publication:
- [P. Bachhav, M. Todisco, and N. Evans, "Latent Representation Learning for Artificial  Bandwidth Extension using a Conditional Variational Auto-Encoder", accepted in ICASSP 2019. ](http://www.eurecom.fr/fr/publication/5817/download/sec-publi-5817.pdf)
____________________________________________________________________________

# Contents

0) Folders "./Speech_files/NB" and "./Speech_files/WB" contain time aligned NB and WB files respectively

1) Feature extraction :
     - This folder should contain features extracted on 'train', 'dev' and 'test' data using the script "Feature_extraction.m" available in folder 	   	     "./../ABE_explicit_memory_ICASSP18/1_Feature_extraction". See "./../ABE_explicit_memory_ICASSP18/Readme.txt" for more details
     - Download our files from [here](https://drive.google.com/drive/folders/1Tj0VtCJygK05B28cbAGsyNyXzetDdqm5?usp=sharing) and copy them to the folder "1_Feature_extraction"

2) Training of variational auto-encoder (VAE) and conditional VAE (CVAE) for feature extraction 
     - Python script "Train_CVAE.py" (available in folder - "2_CVAE_training) can be used to train VAE and CVAE using the data files available in folder "1_Feature_extraction"
     - The keras models are saved in folder "2_CVAE_training/your_models_CVAE" after training 
     - Code successfully tested on - keras with tensorflow (TF) backend (keras - 2.2.4, tensorflow-gpu - 1.12.0 and python - 3.6.7). Note that it is important to train models with TF backend, as models are loaded in MATLAB after training, using MATLAB add-on [importKerasNetwork](https://fr.mathworks.com/help/deeplearning/ref/importkerasnetwork.html;jsessionid=f4ae65d98620b0cc0675f9c3cd38)
      - Script 'Train_SAE.py' trains a conventional stacked auto-encoder (SAE) without pretraining for baseline
      - SSAE baseline can be trained using script [SSAE.py](https://github.com/bachhavpramod/bandwidth_extension/blob/master/ABE_SSAE_IS18/2_SSAE_training/SSAE.py)

3) GMM training :
     - DR is applied to high-dimensional log-spectral data to get lower dimensional features to train a GMM for ABE.
     - Script "Build_GMM_CVAE.m" (available in "3_GMM_training") trains a gaussian mixture model (GMM) for joint vectors obtained using a VAE or a CVAE features and HB LP coefficients. The VAE features are obtained using it's encoder and CVAE features are obtained using it's conditioning network.   
     - The CVAE features for GMM modelling are extracted using the existing trained CVAE models available in folder "2_CVAE_training/models_CVAE"
     - Use the scripts 'Build_GMM_PCA.m', 'Build_GMM_SAE.m' and [Build_GMM_SSAE.m](https://github.com/bachhavpramod/bandwidth_extension/blob/master/ABE_SSAE_IS18/3_GMM_training/Build_GMM_SSAE.m) for DR using principal component analysis (PCA), SAE and SSAE respectively.   

# Demo:
4) Run the scipt "ABE_demo.m" (available in folder - "4_Extension") which demonstrates the ABE using DR techniques, CVAE, SAE, SSAE and PCA. Aseesessment of different artificially extended speech signals is performed using metrics,  RMS-LSD and COSH distance measures and MOS-LQO.
____________________________________________________________________________
# Contact information

For any query, please contact:

- Pramod Bachhav (bachhav at eurecom dot fr)
- Massimiliano Todisco (todisco at eurecom dot fr)
- Nicholas Evans (evans at eurecom dotfr)
____________________________________________________________________________
