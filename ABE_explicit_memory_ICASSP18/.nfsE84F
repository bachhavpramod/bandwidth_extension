Description:
These folders contain MATLAB scripts to perform Artificial Bandwidth Extension (ABE) using explicit memory inclusion along with the two baselines.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Copyright (C) 2018 EURECOM, France.

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

For further details refer to following publication:
P. Bachhav, M. Todisco, and N. Evans, “Exploiting explicit memory inclusion for artificial bandwidth extension,” in Proc. of IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 5459-63, Calgary, Canada.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Usage:

Steps to perform ABE from scratch with your data:

0) Copy time aligned NB and WB files in folders "./Speech_files/NB" and "./Speech_files/WB" respectively

1) Feature extraction :
     - Run the script "Feature_extraction.m" (available in folder "1_Feature_extraction") to extract NB (LogMFE) and HB (LPC) features
     - Use parameters defined in Section 1 (lines 47-48) for the proposed approach and baseline B1 whereas those defined in Section 2 for the baseline B2 (lines 51-52).

2) GMM training :
     - Use the script "Build_GMM_mem_pca.m" (available in folder "2_GMM_training") to train a gaussian mixture model (GMM) for the proposed ABE system with 
       explicit memory inclusion 
     - Use the scipt "Build_GMM.m" to train a GMM for baseline ABE system B1 
     - Use "Build_GMM_mem_delta.m"

3) Run script "ABE_demo1.m" (available in folder "3_Extension") for comparison of ABE using explicit memory with two baselines B1 and B2
     - For this, uncomment the line 'path_to_GMM = './../2_GMM_training/your_models/'
   
   Run script "ABE_demo2.m" to see different plots for comparison/analysis of speech frame of extended signal and corresponding WB speech frame
   
NOTE :
 
You can avoid steps 0), 1) and 2) and Run the script "ABE_demo1.m" OR "ABE_demo2.m" to check ABE outputs with our existing trained models. 

Mutual information:
   - Script "Build_GMM_Calculate_MI.m" (available in folder "MI") trains a GMM and calculates mutual information.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
Contact information
===================

For any query, please contact:

- Pramod Bachhav (bachhav at eurecom dot fr)
- Massimiliano Todisco (todisco at eurecom dot fr)
- Nicholas Evans (evans at eurecom dot fr)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
