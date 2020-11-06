# DurNet
## Action Duration Prediction for Segment-Level Alignment of Weakly-LabeledVideos

Here we describe the content of each folder:

### Alignment_with_TCFPN_PseudoGroundTruth_on_BreakfastDataset: 
This folder contains all the neccessary codes and files to reproduce the results for our model using the TCFPN pseudo ground-truth on the breakfast dataset (Table 1 of the paper). The ReadMe file inside this folder provides further instructions to produce the results of our ablation study too. 
	
	-action2verb.txt: specifies the action indices for each of the defined verbs. 
	-action2object.txt: specifies the corresponding main object of each main action. It also lists the indices for all objects. 




### Alignment_with_NNViterbi_PseudoGroundTruth_on_BreakfastDataset: 
This folder includes codes using both tensorflow (for our method) and pytorch (for NNViterbi). The ReadMe file inside this folder provides further instructions to produce the alignment results using NNViterbi pseudo ground truth on the breakfast dataset. (Table 1 of the paper).
