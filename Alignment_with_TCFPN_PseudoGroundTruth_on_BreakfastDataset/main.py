import numpy as np
import tensorflow as tf
from BeamSearch import BeamSearch
from BeamSearch_Poisson import BeamSearch_Poisson
from LengthModule import LengthModule
import glob
import os
from keras.utils import np_utils
from keras.optimizers import rmsprop
from keras.models import load_model
import metrics
import VisualModel_utils
print (tf.VERSION) # 1.13.1, and Keras version is 2.2.4, PYTHON VERSION 3
dir="./"
var_save_path='./'
num_epochs=41
keep_p=0.895851031640025
hard_label=False
beam_size=150
n_classes=48
n_atomic_actions=14
n_length=7 #output of the length model (# last nodes)
regul=0.0001
obj_regul=0.0001
BG=0  #background index
#exponennts for the action selector module:
zeta_=1     #for the object selector
beta_=40    #for the verb selector
lambda_=1   #for the action recognizer
#params for the TCFPN
n_nodes = [48, 64, 96]
conv_len = 25
vis_n_epoch=100
vis_batch_size=8
# End of params for the TCFPN

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
        gamma[a] = max(int(np.average(step[a])), 1)
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
        if step[a]==0:
            step[a]=1
            # print("step 0 modified")

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

    # assert np.sum(np.asarray(w))==len(labels),'Error in weight calculation'
    w=np.asarray(w)
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

    print('Loading the dataset...Hold On Please...')
    for f_file, t_file in zip(features, transcripts):

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
                action_labels = ignore_repetition(gt_per_frame[5:n:15]) # make sure there is no consecutive repetition

                data[2 * i].append(feature[5:n, 1:])  # feature
                data[2 * i + 1].append(action_labels)  # labels
                gt_per_sec[i].append(gt_per_frame[5:n:15])


    return data,gt_per_sec



data,gt_per_sec=Read_Dataset(dir)
# gt_per_sec=np.load("gt_per_sec.npy")
# data=np.load("data.npy")


print("Data Loaded")


action2atomic_action={0:0,1:2,2:1,3:1,4:3,5:2,6:1,7:1,8:8,9:3,10:1,11:5,12:6,13:7,14:2,15:8,16:2,17:9,18:4,19:2,20:11,21:1,22:2,23:2,24:8,25:3,26:8,27:3,28:1,29:6,30:7,
                      31:1,32:10,33:7,34:12,35:3,36:4,37:2,38:13,39:2,40:7,41:7,42:3,43:1,44:6,45:8,46:1,47:3}


action2object_map={0:18,1:0,2:1,3:2,4:0,5:3,6:4,7:3,8:3,9:3,10:5,11:6,12:5,13:6,14:7,15:7,16:6,17:8,18:9,19:11,20:11,21:11,22:3,23:10,24:12,25:3,26:17,27:0,
28:5,29:5,30:13,31:17,32:9,33:9,34:9,35:0,36:14,37:8,38:10,39:15,40:15,41:14,42:0,43:5,44:5,45:16,46:4,47:3}

for split in [1,2,3,4]:
    ######### Generate the data for this split###############
    #########################################################
    x_train=[]
    y_train=[]
    per_sec_y_train=[]
    x_val=[]
    y_val=[]
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
        pseudo_gt_total = np.load('predictions1.npy') #frame(second)-level pseudo-gt of the baseline (alignment output of their trained model on the training data)
        Vis_model = load_model('Vis_model1_final.h5') #Trained TCFPN model
    if split==2:
        pseudo_gt_total = np.load('predictions2.npy')
        Vis_model = load_model('Vis_model2_final.h5')
    if split==3:
        pseudo_gt_total = np.load('predictions3.npy')
        Vis_model = load_model('Vis_model3_final.h5')
    if split==4:
        pseudo_gt_total = np.load('predictions4.npy')
        Vis_model = load_model('Vis_model4_final.h5')
    pseudo_gt_val=[]
    pseudo_gt = []
    end=0
    pseudo_gt_total = list(pseudo_gt_total)

    pseudo_gt=pseudo_gt_total

    ######### End ###############
    #########################################################


    # Vis_model belongs to Li Ding
    # https://github.com/Zephyr-D/TCFPN-ISBA
    Vis_model.compile(optimizer=rmsprop(lr=1e-6), loss='categorical_crossentropy', sample_weight_mode="temporal")
    ########################################################################
    ########################################################################
    ########################################################################
    print('                                     ')
    x_train_per_sec = [i[::15] for i in x_train]
    x_test_per_sec = [i[::15] for i in x_test]
    x_val_per_sec = [i[::15] for i in x_val]
    max_len = int(np.max([i.shape[0] for i in x_train_per_sec + x_test_per_sec+x_val_per_sec]))
    max_len = int(np.ceil(np.float(max_len) / (2 ** len(n_nodes)))) * 2 **len(n_nodes)
    class_cts = np.array([sum([np.sum(j == i) for j in y_train]) for i in range(n_classes)])
    class_cts = (1 / class_cts) ** 0.5
    class_cts /= (1 / n_classes * np.sum(class_cts))
    class_weight = dict(zip(range(n_classes), class_cts))
    # One-hot encoding
    Y_test = [np_utils.to_categorical(y, n_classes) for y in per_sec_y_test]


    X_test_m, Y_test_m, M_test = VisualModel_utils.mask_data(x_test_per_sec, Y_test, max_len, mask_value=-1)
    VisModelTempPred_test_pre1 = Vis_model.predict(X_test_m[0:], verbose=1)
    VisModelTempPred_test_pre = VisualModel_utils.unmask(VisModelTempPred_test_pre1, M_test[0:])
    prev_meth_pre = [np.argmax(video, axis=1) for video in VisModelTempPred_test_pre] #######################################################################################################################################  ##########  #########
    X_train_m, _, M_train = VisualModel_utils.mask_data(x_train_per_sec, [], max_len, mask_value=-1)
    VisModelTempPred = Vis_model.predict(X_train_m, verbose=1)
    g_1 = tf.Graph()
    with g_1.as_default():
        LengthNet = LengthModule(graph=g_1,nClasses=n_atomic_actions,nActions=n_classes,nObjects=nObjects, length=n_length,
                                 video_info_size=Video_Info_Size,video_obj_info_size=Video_Obj_Info_Size,pre_f_size=64, h_size=64, emb_size=32,num_layers=1, feature_size=64,
                                 batch_size=64, step=3, duration=1*60, sample_rate=4, hard_label=hard_label)

    #####adding video level lebels
    vid_object_info_test=np.squeeze(LengthNet.create_categorical_map_label(action2object_map, y_test, nObjects))
    vid_object_info_tr  = np.squeeze(LengthNet.create_categorical_map_label(action2object_map, y_train, nObjects))

    vid_info_test=np.squeeze(LengthNet.create_categorical_map_label(action2atomic_action, y_test, n_atomic_actions))
    vid_info_tr  =np.squeeze( LengthNet.create_categorical_map_label(action2atomic_action, y_train,n_atomic_actions))
    #####adding video level lebels




    ######### Training the length model #####################
    #########################################################
    print('Training the length model')



    dt_4_length_pred = LengthNet.generate_X_Y(x_train, pseudo_gt,action2atomic_action,action2object_map,True,pseudo_gt,vid_info_tr,vid_object_info_tr)
    dt_4_length_pred_val = LengthNet.generate_X_Y(x_train, per_sec_y_train,action2atomic_action,action2object_map,False,per_sec_y_train,vid_info_tr,vid_object_info_tr)
    X_features, X_actions, Y, N, Sec_Ahead, X_48actions, objects, per_vid_info, per_vid_obj_info = dt_4_length_pred
    data_f_val, data_a_val, labels_val, nSeg_val, Sec_Ahead_val, X_48actions_val, objects_val, per_vid_info_val, per_vid_obj_info_val = dt_4_length_pred_val
    objects_categorical = LengthNet.create_categorical_label( objects, nObjects)
    objects_categorical_val = LengthNet.create_categorical_label(objects_val, nObjects)

    with g_1.as_default():
        LengthNet.model()
    if ft:
        lr=5.062003853160735e-05
        obj_lr=5.062003853160735e-05
    else:
        lr=5.062003853160735e-05
        obj_lr=lr


    min_length_list=FindMinDuration4Video(pseudo_gt)
    length_cap=FindMaxDuration4Actions(X_48actions,N,LengthNet.sample_rate)
    length_cap_sorted = sorted(length_cap.items(), key=lambda s: s[1])
    action2step_map,gamma = FindStep4AtomicActions(Sec_Ahead, n_length, 'median')
    action2bin_size,gamma = FindStep4AtomicActions(Sec_Ahead, n_length, 'mid-median')
    Y = LengthNet.create_adaptive_soft_labels(action2bin_size, Y, n_length,X_actions)  # input Y is int, but the output is the "soft" one-hot vector
    labels_val = LengthNet.create_adaptive_soft_labels(action2bin_size, labels_val, n_length, data_a_val)
    length_class_weight=determine_class_weight(Y,n_length,hard_label)
    base_class_weight = determine_class_weight(X_actions, n_atomic_actions, True)
    object_class_weight = determine_class_weight(objects, nObjects, True)
    N = LengthNet.rescale_N(action2bin_size, N, n_length, X_actions)
    nSeg_val = LengthNet.rescale_N(action2bin_size, nSeg_val, n_length, data_a_val)

    training_pack = [X_features, X_actions, Y, N, X_48actions, objects_categorical, per_vid_info, per_vid_obj_info]
    valid_pack = [data_f_val, data_a_val, labels_val, nSeg_val, X_48actions_val, objects_categorical_val,per_vid_info_val, per_vid_obj_info_val]
    weights=[length_class_weight,base_class_weight,object_class_weight]
    with g_1.as_default():
        LengthNet.train(ft,training_pack,valid_pack,weights,keep_p, var_save_path,num_epochs, lr=lr, regul=regul)
    for XX in [training_pack,valid_pack,X_features,X_actions,Y, N,data_f_val, data_a_val,labels_val, nSeg_val]: #clean up
        del XX


    ft=False


    ########################################################
    ########################################################
    ######## Find the best alignment with beam search #######
    #########################################################
    print('Finding the best alignment with beam search')
    # beam_alg=BeamSearch_Poisson(zeta_,beta_,lambda_)#uncomment for the Poisson Model
    beam_alg = BeamSearch(zeta_,beta_,lambda_) #uncomment for the Duration Network
    VisModelTempPred=VisualModel_utils.unmask(VisModelTempPred,M_train)

    # LAtuples = beam_alg.search(var_save_path, x_train[:], y_train[:], vid_info_tr[:], vid_object_info_tr[:], VisModelTempPred[:],min_length_list, LengthNet, beam_size, length_cap, action2atomic_action,action2object_map, True, action2step_map, action2bin_size,gamma)  #uncomment for the poisson model
    LAtuples = beam_alg.search(var_save_path, x_train, y_train,vid_info_tr,vid_object_info_tr, VisModelTempPred,min_length_list, LengthNet,beam_size,action2atomic_action,action2object_map,True,action2step_map,action2bin_size)  # a list of (lengths,actions) for all videos          #uncomment for the Duration model
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
    #######################################################
    prev_pseudo_gt=prev_pseudo_gt[:]
    per_sec_y_train=per_sec_y_train[:]
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
    np.save("pseudo_gt{}.npy".format(split), pseudo_gt)
    ######### Training the visual model (action Recognizer) #####################
    #############################################################################
    print('Training the visual model (Action Recognizer)')
    print(' ')
    Y_train = [np_utils.to_categorical(y, n_classes) for y in pseudo_gt]
    _, Y_train_m, M_train = VisualModel_utils .mask_data([], Y_train, max_len, mask_value=-1)
    M_train_temp = M_train[:, :, 0]
    for i, j in zip(M_train_temp, pseudo_gt):
        i[:len(j)] += [class_weight[k] for k in j]  # np array can be added with lists
        i[:len(j)] -= 1


    Vis_model = VisualModel_utils.TCFPN(n_nodes, conv_len, n_classes, 64, return_param_str=True, in_len=max_len)
    Vis_model.fit(X_train_m, Y_train_m, epochs=vis_n_epoch, verbose=0, sample_weight=M_train_temp, shuffle=True,batch_size=vis_batch_size)
    Vis_model.save('Vis_model_retrained_split_{}.h5'.format(split))
    VisModelTempPred=Vis_model.predict(X_train_m, verbose=1)
    VisModelTempPred = VisualModel_utils.unmask(VisModelTempPred, M_train)
    RNN_gt = [np.argmax(video, axis=1) for video in VisModelTempPred]
    frame_accuracy2 = np.asarray([np.sum(RNN_gt[i] == per_sec_y_train[i]) / len(RNN_gt[i]) for i in range(len(RNN_gt))])
    fa=np.average(frame_accuracy2)
    print("Segmentation accuracy on the retrained action recognizor on the training data: "+str(fa))




    ################################################
    ######Test Here#################################
    ################################################

    VisModelTempPred_test = Vis_model.predict(X_test_m[0:], verbose=1)
    VisModelTempPred_test = VisualModel_utils.unmask(VisModelTempPred_test, M_test[0:])
    prev_meth = [np.argmax(video, axis=1) for video in VisModelTempPred_test]
    dt_4_length_pred= LengthNet.generate_X_Y(x_train, pseudo_gt, action2atomic_action, action2object_map, False, per_sec_y_train, vid_info_tr,vid_object_info_tr)
    _, _, _, N, _, X_48actions,_,_,_ = dt_4_length_pred
    length_cap = FindMaxDuration4Actions(X_48actions, N, LengthNet.sample_rate)
    print('                                             ')
    print('The Test Stage ::: Allign the test videos::::')
    print('                                             ')
    assert len(y_test)==len(per_sec_y_test)
    TCFPN_test_alligned_seqs = VisualModel_utils.test_allignment(VisModelTempPred_test_pre, per_sec_y_test, y_test)  #stage1
    np.save("TCFPN_test_allignment_original_split{}.npy".format(split),TCFPN_test_alligned_seqs)
    #####
    min_length_list_test = [4 for i in range(len(VisModelTempPred_test))] #Dummy List
    LAtuples = beam_alg.search(var_save_path, x_test, y_test,vid_info_test,vid_object_info_test, VisModelTempPred_test, min_length_list_test, LengthNet, beam_size, action2atomic_action,action2object_map,True,action2step_map,action2bin_size)
    test_results = []
    for (lengths, actions) in LAtuples:  # iterate thru vids
        video_length_sec = np.sum(lengths)
        _y = []
        for L, A in zip(lengths, actions):
            temp = list(A * np.ones(L, dtype=np.int64))
            _y = _y + temp
        _y = np.asarray(_y)
        test_results.append(_y)
    ###Evaluation####-start
    np.save("alignment_test_split{}.npy".format(split), test_results)
    np.save("alignment_TCFPN_test_split{}.npy".format(split), TCFPN_test_alligned_seqs)
    print("Metric for alignemnt-test: ")
    f_acc = metrics.frame_accuracy(test_results, per_sec_y_test, TCFPN_test_alligned_seqs)
    iou=metrics.IoU(list(test_results),per_sec_y_test,BG)
    print("Mine-IOU: " +str(iou))
    iod=metrics.IoD(list(test_results), per_sec_y_test,BG)
    print("Mine-IOD: " + str(iod))
    iou=metrics.IoU(list(TCFPN_test_alligned_seqs),per_sec_y_test,BG)
    print("Previous-IOU: " +str(iou))
    iod=metrics.IoD(list(TCFPN_test_alligned_seqs),per_sec_y_test,BG)
    print("Previous-IOD: " + str(iod))
    metrics.frame_accuracy_w_bg(test_results, per_sec_y_test, TCFPN_test_alligned_seqs)
    print("------------------------------------------------------")
    ##########Evaluation-done
    #####
    print('                                             ')
    print('The Test Stage ::: Segment the test videos:::')
    print('                                             ')



    test_beam_size=50
    ##########Evaluation-start
    print("Metric for segmentation-test: ")
    f_acc = metrics.frame_accuracy(prev_meth, per_sec_y_test, prev_meth_pre)
    iou = metrics.IoU(list(prev_meth), per_sec_y_test,BG)
    print("Mine-IOU: " + str(iou))
    iod = metrics.IoD(list(prev_meth), per_sec_y_test,BG)
    print("Mine-IOD: " + str(iod))
    iou = metrics.IoU(list(prev_meth_pre), per_sec_y_test,BG)
    print("Previous-IOU: " + str(iou))
    iod = metrics.IoD(list(prev_meth_pre), per_sec_y_test,BG)
    print("Previous-IOD: " + str(iod))
    metrics.frame_accuracy_w_bg(prev_meth, per_sec_y_test, prev_meth_pre)
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    ##########Evaluation-done

print("Done")


