import numpy as np
import tensorflow as tf
from BeamSearch import BeamSearch
from LengthModule import LengthModule
from index_conversion import index_conversion
import glob
import os
from keras.utils import np_utils
import metrics
from sklearn.utils.class_weight import compute_class_weight
import argparse
print (tf.VERSION) #1.2.1   #1.13.1  , keras version is 2.2.4
dir="./"
test_mode='True'   #if True, it runs the test, and if False, it generates new pg on the training data
var_save_path='./'
dict_word_to_index_file_name='./action_object_list.txt'
num_epochs=91
keep_p=0.895851031640025
hard_label=False
beam_size=150
n_classes=48
n_atomic_actions=14
n_length=7 #output of the length model (# last nodes)
regul=0.0001
obj_regul=0.0001
BG=0  #background index
n_nodes = [48, 64, 96]
conv_len = 25
#####
zeta_=5
beta_=30
lambda_=1
#####
nObjects=19
Video_Info_Size=n_atomic_actions#13
Video_Obj_Info_Size=nObjects#18
def FindStep4AtomicActions(Sec_Ahead,nlength_bin,mode):
    step = {}
    gamma={}
    for a, l in Sec_Ahead:
        L = l  #
        if a in step:
            step[a].append(L)
        else:
            step[a]=[]
            step[a].append(L)
    for a in step:
        gamma[a] = max(int(np.average(step[a])), 5)
        if mode=='mid-median':  #median is the middle bin
            step[a] = int(np.median(step[a]) // ((nlength_bin // 2) + 1))
        elif mode=='median':      #median is the max
            step[a]=int(  np.median(step[a]) // nlength_bin )
        elif mode == 'average':   #average is the max
            step[a] = int(np.average(step[a]) // nlength_bin)
        elif mode=='maximum':      #max is the max length
            step[a] = int(np.max(step[a]) // nlength_bin)
        else:
            assert 1==0," mode not specified!!!!!"

    return step,gamma


def map_action2label(action_obj_list):
    action2label_dict={}
    with open(action_obj_list, 'r') as action_obj_file:
            for (i, line) in enumerate(action_obj_file):
                item= line.split(' ')
                if item[1] not in action2label_dict:
                    action2label_dict[item[1]]=[item[0],item[2],item[3],item[4][:-1]]


    return action2label_dict

def FindMinDuration4Video(pseudo_gt):
    min_length=[]
    for video in pseudo_gt:
        temp = video[1:] - video[0:-1]
        idx = np.where(temp != 0)
        if np.sum(idx)==0:
            min_length.append(len(video))
            continue
        idx = idx[0] + 1
        idx = np.append(idx, len(video))
        MIN=np.min(idx[1:]-idx[0:-1])
        MIN=min(MIN,idx[0])
        min_length.append(MIN)
    return min_length

def FindMaxDuration4Actions(X_actions,N,sample_rate):
    length_cap = {}
    for a, l in zip(X_actions, N):
        L=(l+1)*sample_rate #
        if a in length_cap:
            if L> length_cap[a]:
                length_cap[a] =L
        else:
            length_cap[a] = L
    return length_cap

def ignore_repetition(actions):
    actions = np.asarray(actions)
    temp = actions[1:] - actions[0:-1]
    idx = np.where(temp != 0)
    if np.sum(idx)==0:
        u_action=np.asarray([actions[-1]])
        return u_action
    u_action = actions[idx]
    if len(actions != 0):
        u_action = np.append(u_action, actions[-1])

    return u_action

def find_length_stat(gt):
    MIN = 10000
    guilty=[]
    length_dic = {}
    for split in gt:
        for vid in split:
            counter = 1
            for i in range(len(vid)):
                if vid[i]==0:
                    continue
                if i + 1 == len(vid):
                    if counter<MIN:
                        MIN=counter
                    if counter not in length_dic:
                        length_dic[counter]=1
                    else:
                        length_dic[counter]=length_dic[counter]+1
                    if counter==2:
                        guilty.append(vid[i])
                    break
                if vid[i] == vid[i + 1]:
                    counter = counter + 1
                else:
                    if counter<MIN:
                        MIN=counter
                    if counter not in length_dic:
                        length_dic[counter]=1
                    else:
                        length_dic[counter]=length_dic[counter]+1
                    if counter==2:
                        guilty.append(vid[i])
                    counter = 1
    return MIN,length_dic,guilty


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([1, number_dim, 2])
    for i in range(number_dim):
        weights[0, i, :] = compute_class_weight('balanced', [0, 1], y_true[:, i])
    return weights


def determine_class_weight(labels,nclasses,hard):
    #labels : [#samples], assuming 0<=labels[i]<nclasses
    w=[x for x in [0]*nclasses]
    for i in range(len(labels)):
        if type(labels[i])==list:
            for j in range(len(labels[i])):
                if not hard:
                    klasse=np.argmax(labels[i][:])
                else:
                    klasse = labels[i][j]
                w[klasse]=w[klasse]+1
        else:
            if not hard:
                klasse = np.argmax(labels[i][:])
                w[klasse] = w[klasse] + 1
            elif labels[i]<nclasses:
                klasse = labels[i]
                w[klasse] = w[klasse] + 1


    w=np.asarray(w)+1
    class_cts = (1 / w) ** 0.5
    class_cts /= (1 / nclasses * np.sum(class_cts))
    # print("The distribution of classes for the training set: ")
    # print(w/np.sum(w))
    return class_cts

def Read_Dataset(dir):
    dict_word_to_index_file_name = 'action_object_list.txt'
    action2label =map_action2label(dict_word_to_index_file_name)
    data = [[] for i in range(8)]  # x,y for 4 splits
    gt_per_sec=[[] for i in range(4)]
    all = ["P%02d" % i for i in range(3, 56)]
    splits = [all[:13], all[13:26], all[26:39], all[39:52]]
    features = glob.glob(os.path.join(dir,'features','*.txt'))
    transcripts = glob.glob(os.path.join(dir, 'labels','*.txt'))
    features.sort()
    transcripts.sort()
    id=0
    print('Loading the dataset...Hold On Please...')
    for f_file, t_file in zip(features, transcripts):
        print(id)
        id=id+1
        person=t_file.split('_')[0].split('/')[-1]
        for i, split in enumerate(splits):
            if person in split:

                feature =np.loadtxt(f_file)


                actions = open(t_file).readlines()
                gt_per_frame = np.repeat(0, int(actions[-1].split()[0].split('-')[1]))
                for act in actions:
                    tm, lb = act.split()
                    gt_per_frame[(int(tm.split('-')[0]) - 1):int(tm.split('-')[1])] = int(action2label[lb][0])

                n = min(len(feature), len(gt_per_frame)) # make sure the same length
                action_labels = ignore_repetition(gt_per_frame[5:n:]) # make sure there is no consecutive repetition


                data[2 * i].append(feature[5:n, 1:])  # feature
                data[2 * i + 1].append(action_labels)  # video level labels
                gt_per_sec[i].append(gt_per_frame[5:n:]) # frame level labels


    return data,gt_per_sec


parser = argparse.ArgumentParser()
parser.add_argument("t_mode", help="Is this for test mode ? True or False?")
args = parser.parse_args()
test_mode=args.t_mode
print(test_mode)
print(' ')
if test_mode=='True':
    print("Code runs in test mode:")
else:
    print("Code runs in training mode ----> create new pseudo ground-truth:")

print(' ')

# data,gt_per_sec=Read_Dataset(dir)
gt_per_sec=np.load("gt_per_sec.npy")
data=np.load("data.npy")
gt_per_sec[3][54]=np.concatenate([gt_per_sec[3][54],np.zeros((10),dtype=int)])
data[6][54]=np.concatenate([data[6][54],np.tile(data[6][54][-1,:], (10, 1))],axis=0)

print("Dataset Loaded!")

#based on action2verb.txt
action2atomic_action={0:0,1:2,2:1,3:1,4:3,5:2,6:1,7:1,8:8,9:3,10:1,11:5,12:6,13:7,14:2,15:8,16:2,17:9,18:4,19:2,20:11,21:1,22:2,23:2,24:8,25:3,26:8,27:3,28:1,29:6,30:7,
                      31:1,32:10,33:7,34:12,35:3,36:4,37:2,38:13,39:2,40:7,41:7,42:3,43:1,44:6,45:8,46:1,47:3}


#based on action2object.txt
action2object_map={0:18,1:0,2:1,3:2,4:0,5:3,6:4,7:3,8:3,9:3,10:5,11:6,12:5,13:6,14:7,15:7,16:6,17:8,18:9,19:11,20:11,21:11,22:3,23:10,24:12,25:3,26:17,27:0,
28:5,29:5,30:13,31:17,32:9,33:9,34:9,35:0,36:14,37:8,38:10,39:15,40:15,41:14,42:0,43:5,44:5,45:16,46:4,47:3}

act2actmap={}
actions = open("act2actmap.txt").readlines()
for i in range(48):  #mapping my labels to baseline labels; the indices used in NNviterbi are different than mine so they need to mapped later
    act2actmap[int(actions[i].split()[1])]=int(actions[i].split()[0])
print(' ')
print("Converting indices of the initial pseudo-gt...!")
print(' ')
Conversion=index_conversion()
Conversion.NNViterbi_to_Ours() #convert the index encoding used in NNViterbi to the ones used here
if test_mode=='True':
    Conversion.create_GRU_test()

for split in [1,2,3,4]:
    round=0
    print("beam size is : "+str(beam_size))
    while(round<1):
        round=round+1
        ######### Generate the data for this split###############
        #########################################################
        x_train=[]
        y_train=[]
        per_sec_y_train=[]
        x_val=[]
        y_val=[]
        per_sec_y_val=[]
        ft=True
        print("Split Number: "+ str(split))
        for i in [1,2,3,4]:
            if i!=split:
                x_train=x_train+data[(i-1)*2]
                y_train=y_train+data[(i-1)*2+1]
                per_sec_y_train =per_sec_y_train+ gt_per_sec[i - 1]

        x_test=data[(split-1)*2]
        y_test=data[(split-1)*2+1]



        per_sec_y_test = gt_per_sec[split - 1]
        if split==1:
            pseudo_gt_total = np.load('predictions1.npy')  #frame-level pseudo-gt of the baseline (alignment output of their trained model on the training data)
            VisModelTempPred = np.load("weak_predictions1.npy") #frame-level softmax values of the action recognizer(before being retrained) on the training data, required for generating new pg
            if test_mode=='True':
                VisModelTempPred_test =np.load("GRU_split1_test.npy") #frame-level softmax values of the action recognizer(after being retrained) on the test data,
        if split==2:
            pseudo_gt_total = np.load('predictions2.npy')
            VisModelTempPred = np.load("weak_predictions2.npy")
            if test_mode=='True':
                VisModelTempPred_test = np.load("GRU_split2_test.npy")
        if split==3:
            pseudo_gt_total = np.load('predictions3.npy')
            VisModelTempPred = np.load("weak_predictions3.npy")
            if test_mode=='True':
                VisModelTempPred_test = np.load("GRU_split3_test.npy")
        if split==4:
            pseudo_gt_total = np.load('predictions4.npy')
            VisModelTempPred = np.load("weak_predictions4.npy")
            if test_mode=='True':
                VisModelTempPred_test = np.load("GRU_split4_test.npy")
        pseudo_gt_val=[]
        pseudo_gt = []
        end=0
        pseudo_gt_total = list(pseudo_gt_total)
        acc=[]
        for vid in per_sec_y_test:
            bg=vid==0
            acc.append(np.sum(bg)/len(vid))
        print(str(np.average(acc))+" in average BG percentage")
        pseudo_gt=pseudo_gt_total



        x_train_per_sec = [i[::] for i in x_train]
        x_test_per_sec = [i[::] for i in x_test]
        x_val_per_sec = [i[::] for i in x_val]
        max_len = int(np.max([i.shape[0] for i in x_train_per_sec + x_test_per_sec]))
        max_len = int(np.ceil(np.float(max_len) / (2 ** len(n_nodes)))) * 2 **len(n_nodes)

        # One-hot encoding
        Y_test = [np_utils.to_categorical(y, n_classes) for y in per_sec_y_test]


        g_1 = tf.Graph()
        with g_1.as_default():
            LengthNet = LengthModule(graph=g_1,nClasses=n_atomic_actions,nActions=n_classes,nObjects=nObjects, length=n_length,
                                     video_info_size=Video_Info_Size,video_obj_info_size=Video_Obj_Info_Size,pre_f_size=1*64, h_size=1*64, emb_size=1*32,num_layers=1, feature_size=64,
                                     batch_size=64, step=3, duration=60, sample_rate=15*4, hard_label=hard_label)

        #####adding video level lebels
        vid_object_info_test=np.squeeze(LengthNet.create_categorical_map_label(action2object_map, y_test, nObjects))
        vid_object_info_tr  = np.squeeze(LengthNet.create_categorical_map_label(action2object_map, y_train, nObjects))

        vid_info_test=np.squeeze(LengthNet.create_categorical_map_label(action2atomic_action, y_test, n_atomic_actions))
        vid_info_tr  =np.squeeze( LengthNet.create_categorical_map_label(action2atomic_action, y_train,n_atomic_actions))
        #####adding video level lebels

        loop=0


        while(loop<1):
            loop=loop+1
            print("loop " + str(loop))
            ######### Training the length model #####################
            #########################################################
            print('Training the length model')

            dt_4_length_pred = LengthNet.generate_X_Y(x_train, pseudo_gt,action2atomic_action,action2object_map,True,pseudo_gt,vid_info_tr,vid_object_info_tr)
            dt_4_length_pred_val = LengthNet.generate_X_Y(x_test, per_sec_y_test,action2atomic_action,action2object_map,False,per_sec_y_test,vid_info_test,vid_object_info_test)
            X_features, X_actions, Y, N,Sec_Ahead, X_48actions, objects,per_vid_info,per_vid_obj_info=dt_4_length_pred
            data_f_val, data_a_val, labels_val, nSeg_val,Sec_Ahead_val, X_48actions_val,objects_val,per_vid_info_val,per_vid_obj_info_val=dt_4_length_pred_val
            objects_categorical = LengthNet.create_categorical_label( objects, nObjects)
            objects_categorical_val = LengthNet.create_categorical_label(objects_val, nObjects)
            if loop==1:
                with g_1.as_default():
                    LengthNet.model()
            if ft:
                lr=5.062003853160735e-05
                obj_lr=5.062003853160735e-05
            else:
                lr=5.062003853160735e-05
                obj_lr=lr
                num_epochs=91

            min_length_list=FindMinDuration4Video(pseudo_gt)
            length_cap=FindMaxDuration4Actions(X_48actions,N,LengthNet.sample_rate)
            length_cap_sorted = sorted(length_cap.items(), key=lambda s: s[1])
            action2step_map, gamma = FindStep4AtomicActions(Sec_Ahead, n_length, 'median')
            action2bin_size, gamma = FindStep4AtomicActions(Sec_Ahead, n_length, 'mid-median')
            Y = LengthNet.create_adaptive_soft_labels(action2bin_size, Y, n_length,X_actions)  # input Y is int, but the output is the "soft" one-hot vector
            labels_val = LengthNet.create_adaptive_soft_labels(action2bin_size, labels_val, n_length, data_a_val)
            length_class_weight=determine_class_weight(Y,n_length,hard_label)
            base_class_weight = determine_class_weight(X_actions, n_atomic_actions, True)
            object_class_weight = determine_class_weight(objects, nObjects, True)
            N= LengthNet.rescale_N(action2bin_size, N, n_length, X_actions)
            nSeg_val = LengthNet.rescale_N(action2bin_size, nSeg_val, n_length,data_a_val)

            training_pack=[X_features,X_actions,Y,N,X_48actions,objects_categorical,per_vid_info,per_vid_obj_info]
            valid_pack=[data_f_val,data_a_val,labels_val,nSeg_val, X_48actions_val,objects_categorical_val,per_vid_info_val,per_vid_obj_info_val]
            weights=[length_class_weight,base_class_weight,object_class_weight]
            with g_1.as_default():
                LengthNet.train(ft,training_pack,valid_pack,weights,keep_p, var_save_path,num_epochs, lr=lr, regul=regul,object_lr=obj_lr)
            for XX in [training_pack,valid_pack,X_features,X_actions,Y, N,data_f_val, data_a_val,labels_val, nSeg_val]: #clean up
                del XX


            ft=False


            ########################################################
            ########################################################


            ######## Find the best alignment with beam search #######
            #########################################################
            print('Finding the best alignment with beam search')
            beam_alg = BeamSearch(zeta_,beta_,lambda_)
            if test_mode=='True':
                continue
            LAtuples = beam_alg.search(var_save_path, x_train, y_train,vid_info_tr,vid_object_info_tr, VisModelTempPred,min_length_list, LengthNet,beam_size,action2atomic_action,action2object_map,True,act2actmap,action2step_map,action2bin_size)  # a list of (lengths,actions) for all videos
            prev_pseudo_gt=pseudo_gt
            pseudo_gt=[]
            for (lengths,actions) in LAtuples: #iterate thru vids
                video_length_sec=np.sum(lengths)
                _y=[]
                for L,A in zip(lengths,actions):
                    temp = list(A*np.ones(L,dtype=np.int64))
                    _y=_y+temp
                _y=np.asarray(_y)
                pseudo_gt.append(_y)
            ########################################################
            ########################################################
            #########################################################
            np.save("pseudo_gt_{}.npy".format(split), pseudo_gt)
            t = [np.sum(pseudo_gt[i] != prev_pseudo_gt[i]) for i in range(len(pseudo_gt))]
            T = [len(pseudo_gt[i]) for i in range(len(pseudo_gt))]
            diff_percentage = np.sum(t) / np.sum(T)
            # diff_percentage=[np.sum(pseudo_gt[i]!=prev_pseudo_gt[i])/len(pseudo_gt[i]) for i in range(len(pseudo_gt))]
            print("The average frame diff percentage is " + str(np.average(diff_percentage)))
            ########################################################
            ########################################################
            f_acc=metrics.frame_accuracy(pseudo_gt, per_sec_y_train,prev_pseudo_gt)
            metrics.frame_accuracy_w_bg(pseudo_gt, per_sec_y_train, prev_pseudo_gt)
            iou = metrics.IoU(list(pseudo_gt), per_sec_y_train,BG)
            print("Mine-IOU: " + str(iou))
            iod = metrics.IoD(list(pseudo_gt), per_sec_y_train,BG)
            print("Mine-IOD: " + str(iod))
            iou = metrics.IoU(list(prev_pseudo_gt), per_sec_y_train,BG)
            print("Prev-IOU: " + str(iou))
            iod = metrics.IoD(list(prev_pseudo_gt), per_sec_y_train,BG)
            print("Prev-IOD: " + str(iod))


    if test_mode!='True':
        continue

    #
    # #####################   ########    ###
    # ######Test Here######   ########    ###
    # #####################   ########    ###

    prev_meth = [np.argmax(video, axis=1) for video in VisModelTempPred_test]

    min_length_list_test = [4 for i in range(len(VisModelTempPred_test))] #Dummy List
    LAtuples = beam_alg.search(var_save_path, x_test, y_test,vid_info_test,vid_object_info_test, VisModelTempPred_test, min_length_list_test, LengthNet, beam_size, action2atomic_action,action2object_map,True,act2actmap, action2step_map, action2bin_size)
    test_results = []
    np.save("LAtuples_test_{}.npy".format(split),LAtuples)
    for (lengths, actions) in LAtuples:  # iterate thru vids
        video_length_sec = np.sum(lengths)
        _y = []
        for L, A in zip(lengths, actions):
            temp = list(A * np.ones(L, dtype=np.int64))
            _y = _y + temp
        _y = np.asarray(_y)
        test_results.append(_y)
    ###Evaluation####-start
    np.save("alignment_test_{}.npy".format(split), test_results)
    print("Metrics for alignemnt-test: ")
    f_acc = metrics.frame_accuracy(test_results, per_sec_y_test, test_results)
    iou=metrics.IoU(list(test_results),per_sec_y_test,BG)
    print("Mine-IOU: " +str(iou))
    iod=metrics.IoD(list(test_results), per_sec_y_test,BG)
    print("Mine-IOD: " + str(iod))
    iou=metrics.IoU(list(test_results),per_sec_y_test,BG)
    print("Previous-IOU: " +str(iou))
    iod=metrics.IoD(list(test_results),per_sec_y_test,BG)
    print("Previous-IOD: " + str(iod))
    metrics.frame_accuracy_w_bg(test_results, per_sec_y_test,test_results)
    print("------------------------------------------------------")
    ##########Evaluation-done
    #####
    print('                                             ')
    print('The Test Stage ::: Segment the test videos:::')
    print('                                             ')
    #
    #
        ##########Evaluation-start
    print("Metrics for GRU : ")
    f_acc = metrics.frame_accuracy(prev_meth, per_sec_y_test, prev_meth)
    iou = metrics.IoU(list(prev_meth), per_sec_y_test,BG)
    print("Mine-IOU: " + str(iou))
    iod = metrics.IoD(list(prev_meth), per_sec_y_test,BG)
    print("Mine-IOD: " + str(iod))
    iou = metrics.IoU(list(prev_meth), per_sec_y_test,BG)
    print("Previous-IOU: " + str(iou))
    iod = metrics.IoD(list(prev_meth), per_sec_y_test,BG)
    print("Previous-IOD: " + str(iod))
    metrics.frame_accuracy_w_bg(prev_meth, per_sec_y_test, prev_meth)
    print("------------------------------------------------------")
        ##########Evaluation-done
if test_mode != 'True':
    Conversion.Ours_to_NNViterbi()  # convert our index encoding to NNViterbi's
print("Done")

