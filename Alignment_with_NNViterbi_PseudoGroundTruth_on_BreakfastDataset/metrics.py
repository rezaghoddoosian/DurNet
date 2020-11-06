import numpy as np


def frame_accuracy(test,gt,prev_pseudo_gt):
    t = [np.sum(test[i] == gt[i])  for i in range(len(test))]
    T = [len(test[i]) for i in range(len(test))]
    frame_accuracy=np.sum(t)/np.sum(T)
    t2 = [np.sum(prev_pseudo_gt[i] == gt[i])  for i in range(len(test))]
    frame_accuracy2=np.sum(t2)/np.sum(T)
    per_vid_frame_accuracy1 = np.asarray([np.sum(test[i] == gt[i]) / len(test[i]) for i in range(len(test))])
    per_vid_frame_accuracy2 = np.asarray([np.sum(prev_pseudo_gt[i] == gt[i]) / len(prev_pseudo_gt[i]) for i in range(len(prev_pseudo_gt))])
    print("previously:")
    print(np.average(per_vid_frame_accuracy2))
    print("#####")
    print("Currently:")
    print(np.average(per_vid_frame_accuracy1))

    return frame_accuracy

def fine_proximity_accuracy(err_distance):
    proximity2 = []
    for idx in range(np.shape(err_distance)[0]):
        if err_distance[idx] <= 1:
            proximity2.append(1)
        else:
            proximity2.append(0)
    return proximity2

def proximity_accuracy(proximity2,length,err_distance,true_label,hard):

    proximity1=[]
    err_distance=err_distance*5
    for idx in range(np.shape(err_distance)[0]):
        a = (3.0 / (length*5 - 1))
        b = a - 1.0
        if hard:
            ind = true_label[idx]
            true=3+ind*5
        else:
            ind=np.argmax(true_label[idx][:])
            true = 3 + ind * 5
        sigma = (a * (true)) - b
        sigma=5
        if err_distance[idx] <= sigma * 2:
            proximity2.append(1)
        else:
            proximity2.append(0)
        if err_distance[idx] <= sigma * 1:
            proximity1.append(1)
        else:
            proximity1.append(0)
    return proximity1,proximity2


def per_class_acc(true,dist,nclass,hard):  # TP/#Positives  #Recall
    acc=[]
    if hard==False:
        argmax = np.argmax(true, axis=1)
    else:
        argmax=true
    for i in range(nclass):
        inds=argmax==i
        correct=np.sum(dist[inds]==0)
        acc.append(correct/np.sum(inds))
    return acc

def per_class_prec_vector(true,nclass,pred):  # TP/#Positives  #Recall
    acc=[]
    argmax_t = np.argmax(true, axis=1)
    argmax_p = np.argmax(pred, axis=1)
    for i in range(nclass):
        inds=argmax_p==i
        inds2 = argmax_t == i
        correct =np.sum(inds*inds2)
        acc.append(correct/np.sum(inds))
    return acc

def per_class_prec(true,pred,gt,n_acts): #TP/#Classified as T
        prec={}
        PREC=[]

        for i,true_video in enumerate(true):

            u_gt=np.unique(gt[i])
            for action in u_gt:
                TOTAL_temp=np.sum(pred[i]==action)
                tp_temp=np.sum(pred[i][true_video==action]==action)
                if action in prec:
                    prec[action].append(tp_temp/TOTAL_temp)
                else:
                    prec[action]=[tp_temp/TOTAL_temp]



        for act in range(n_acts):
            PREC.append(np.average(prec[act]))
            # sorted(prec.items(), key=lambda s: s[0])
        return np.asarray(PREC)


def per_class_acc1(true, pred, gt, n_acts):  # TP/#Positives  #Recall
    prec = {}
    PREC = []

    for i, true_video in enumerate(true):

        u_gt = np.unique(gt[i])
        for action in u_gt:
            TOTAL_temp = np.sum(true[i] == action)
            tp_temp = np.sum(pred[i][true_video == action] == action)
            if action in prec:
                prec[action].append(tp_temp / TOTAL_temp)
            else:
                prec[action] = [tp_temp / TOTAL_temp]

    for act in range(n_acts):
        PREC.append(np.average(prec[act]))
        # sorted(prec.items(), key=lambda s: s[0])
    return np.asarray(PREC)

def IoU(P, Y, bg_class=None):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager
    # ICRA 2016
    def segment_labels(Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
        return Yi_split

    def segment_intervals(Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        return intervals

    def overlap_(p, y, bg_class):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array([t for t, l in zip(true_intervals, true_labels) if l != bg_class])
            true_labels = np.array([l for l in true_labels if l != bg_class])
            pred_intervals = np.array([t for t, l in zip(pred_intervals, pred_labels) if l != bg_class])
            pred_labels = np.array([l for l in pred_labels if l != bg_class])

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, np.float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i] == pred_labels[j]:
                    intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0],
                                                                                         true_intervals[i][0])
                    union = max(pred_intervals[j][1], true_intervals[i][1]) - min(pred_intervals[j][0],
                                                                                  true_intervals[i][0])
                    score_ = float(intersection) / union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean() * 100

    if type(P) == list:
        return np.mean([overlap_(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_(P, Y, bg_class)


def IoD(P, Y, bg_class=None):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager
    # ICRA 2016

    def segment_labels(Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
        return Yi_split

    def segment_intervals(Yi):
        idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
        intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        return intervals

    def overlap_d(p, y, bg_class):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array([t for t, l in zip(true_intervals, true_labels) if l != bg_class])
            true_labels = np.array([l for l in true_labels if l != bg_class])
            pred_intervals = np.array([t for t, l in zip(pred_intervals, pred_labels) if l != bg_class])
            pred_labels = np.array([l for l in pred_labels if l != bg_class])

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, np.float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i] == pred_labels[j]:
                    intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0],
                                                                                         true_intervals[i][0])
                    union = pred_intervals[j][1] - pred_intervals[j][0]
                    score_ = float(intersection) / union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean() * 100

    if type(P) == list:
        return np.mean([overlap_d(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_d(P, Y, bg_class)


def action_accuracy(p_actions,g_actions):#The reference is the gt. What percentage of the gt labels are predicted correctly?
    def ignore_repetition(actions):
        actions = np.asarray(actions)
        temp = actions[1:] - actions[0:-1]
        idx = np.where(temp != 0)
        if len(idx) == 0:
            u_action = np.asarray(actions[-1])
            return u_action
        u_action = actions[idx]
        if len(actions != 0):
            u_action = np.append(u_action, actions[-1])
        return u_action

    def sub_action_accuracy(p_actions,g_actions):   #for 1 video, for each video evaluates the percentage of correctly detected action labels disregarding time_stamps
        p_actions=ignore_repetition(p_actions)
        scores = np.zeros(len(g_actions), np.float)
        for i in range(len(g_actions)):
            for j in range(len(p_actions)):
                if g_actions[i]==p_actions[j]:
                    scores[i]=1
                    p_actions[j]=-1
                    break

        return scores.mean()




    return sub_action_accuracy(p_actions, g_actions)


def action_accuracy2(p_actions,g_actions): #The reference is the predicted labels. What percentage of the predicted labels are present in the video ?
    def ignore_repetition(actions):
        actions = np.asarray(actions)
        temp = actions[1:] - actions[0:-1]
        idx = np.where(temp != 0)
        if len(idx) == 0:
            u_action = np.asarray(actions[-1])
            return u_action
        u_action = actions[idx]
        if len(actions != 0):
            u_action = np.append(u_action, actions[-1])
        return u_action

    def sub_action_accuracy(p_actions,g_action):   #for 1 video, fpr each video evaluates the percentage of correctly detected action labels disregarding time_stamps
        g_actions = np.copy(g_action)
        p_actions=ignore_repetition(p_actions)
        scores = np.zeros(len(p_actions), np.float)
        for i in range(len(p_actions)):
            for j in range(len(g_actions)):
                if g_actions[j]==p_actions[i]:
                    scores[i]=1
                    g_actions[j]=-1
                    break

        return scores.mean()




    return sub_action_accuracy(p_actions, g_actions)

def print2file_accuracy_for_categorical_output(prob,lb,n,threshold,filename):
    #prob : [#samples,nClasses]
    e=0.000000000000000000001
    prob[prob>=threshold]=1
    prob[prob < threshold] = 0
    perclass_tp=np.sum(prob*lb,axis=0)
    perclass_p=np.sum(prob,axis=0)
    perclass_t=np.sum(lb,axis=0)

    precision=perclass_tp/(perclass_p+e) #how many of my predictions are correct
    recall=perclass_tp/(perclass_t+e)    #how many of the items are being predicted
    with open(filename, 'a') as f:
        f.write(str(precision))
        f.write("\n  ")
        f.write(str(n)+" ) ---> Up Precision  *******  Below Recall   ")
        f.write("\n  ")
        f.write(str(recall))
        f.write("\n  ")
        f.write("\n  ")
    return


def print2file_accuracy_for_one_hot_output(acc,prob,n,training,nclass,true,filename):
    with open(filename, 'a') as f:
        if training:
            f.write("\n  ")
            f.write(" ################### NEW ROUND ###################  ")
            f.write("\n  ")
            f.write("Base: Training|| Epoch " + str(n) + " :" + " and accuracy is: " + str(acc))
            f.write("\n  ")
        else:
            f.write("\n  ")
            f.write(" ################### NEW ROUND ###################  ")
            f.write("\n  ")
            f.write("Base: Test|| Epoch " + str(n) + " :" + " and accuracy is: " + str(acc))
            q = np.argmax(prob, 1)
            qh = np.bincount(q)
            f.write("\n  ")
            f.write(str(qh / np.sum(qh)))
            f.write("\n  ")

            acc = []
            argmax_t = np.argmax(true, axis=1)
            argmax_p = np.argmax(prob, axis=1)
            for i in range(nclass):
                inds = argmax_p == i
                inds2 = argmax_t == i
                correct = np.sum(inds * inds2)
                acc.append(correct / (np.sum(inds)+1e-20) )

            f.write(str(acc))
            f.write("\n  ")
            f.write("\n  ")


    return

def frame_accuracy_w_bg(test,gt,prev_pseudo_gt):
    per_vid_frame_accuracy1=[]
    per_vid_frame_accuracy2 = []
    for i in range(len(test)):
        gt_ind= gt[i] != 0
        per_vid_frame_accuracy1.append(np.sum(test[i][gt_ind] == gt[i][gt_ind]) / len(gt[i][gt_ind]))
        per_vid_frame_accuracy2.append(np.sum(prev_pseudo_gt[i][gt_ind] == gt[i][gt_ind]) / len(gt[i][gt_ind]))

    per_vid_frame_accuracy1 = np.asarray(per_vid_frame_accuracy1)
    per_vid_frame_accuracy2 = np.asarray(per_vid_frame_accuracy2)
    acc=np.mean(per_vid_frame_accuracy1)
    prev_acc = np.mean(per_vid_frame_accuracy2)
    print(" ")
    print("-> acc_wo_bg: current acc is "+str(acc)+" and the previous was "+str(prev_acc))
    print(" ")
    return