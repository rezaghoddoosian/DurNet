Main Software Requirements:
tensorflow version: 1.13
keras version: 2.2.4
Python version 3
Pytorch 1.1
Ubuntu 16


********************************************************************************************************************
Instructions to reproduce the results denoted as "ours/[28] pg" on Table 1, for the Breakfast dataset.
These results were obtaiend using NNViterbi pseudo ground-truth (pg) for "alignment":
********************************************************************************************************************

******Note: This process includes running codes using both tensorflow (for our method) and pytorch (for NNViterbi), and also exchanging data between these two environments. The following steps are written assuming each environment's independence from the other one, so the requirements of both envoronements can be fullfilled seperately.


####### Disclaimer ######################################################################################################################################
The codes to run the NNViterbi alignment method are obtained from https://github.com/alexanderrichard/NeuralNetwork-Viterbi and modified accordingly.
#########################################################################################################################################################











#######Data Preparation for NNViterbi#########################################################################
#############################################################################################################
0-Download the data from https://uni-bonn.sciebo.de/s/vVexqxzKFc6lYJx
0-1-Extract it so that you have the "data" folder in the same directory as train_stage1.py
0-2-The filenames of all 4 splits (train and test) are provided in the "split_fnames" folder. Copy those files in the "data" folder.

######Data Preparation for Our Method########################################################################
#############################################################################################################
1- Download the breakfast dataset from http://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/
1-1- The data used in the code is Frame-based precomputed reduced FV (64 dim): breakfast_data.tar.gz (~1GB) and Coarse segmentation information: segmentation_coarse.tar.gz
1-2- Make sure all feature files (.txt), that were downloaded from the breakfast dataset, are stored in one folder named : "features". Make this folder a subfolder of the directory where our .py files are stored.
1-3- Make sure all label files (.txt), that were downloaded from the breakfast dataset, are stored in one folder named : "labels". Make this folder a subfolder of the directory where our .py files are stored.
1-4- Make sure 'action_object_list.txt' and 'act2actmap.txt' are in the same directory as the other .py files

############################### Training-Stage1A- Create Pseudo Ground-truth ################################
#############################################################################################################
2-Run the shell script:   run_align_exp_stage1.sh
The above line trains a model based on the logic of NNViterbi and produces pseudo ground truth results and GRU softmax values for the training set of all 4 splits. This stage is run in pytorch and it creates the neccessary folders for the prediction files. This step corresponds to training of the NNViterbi method.

############################### Training-Stage1B- Create New Pseudo Ground-truth ############################
#############################################################################################################
3- Train our method by using NNViterbi's pseudo groundtruths and GRU Softmax values of the training data. This code is developed in Tensorflow. In order to do so, just type the following command line in terminal:
python main.py False
############################### Training-Stage2- Retrain the GRU ############################################
#############################################################################################################
4- Train the NNViterbi method for one more epoch (it converges after one epoch) using our "new pseudo ground-truth". To do so, type the following shell script:  run_align_exp_stage2.sh 
############################### Test- Alignment #############################################################
#############################################################################################################
5- Type the following command in terminal: python main.py True
This line does the segment-level beam search of our method for alignment. These results correspond to the results in Table 1 of the paper denoted as "Ours/[28]pg". 





















