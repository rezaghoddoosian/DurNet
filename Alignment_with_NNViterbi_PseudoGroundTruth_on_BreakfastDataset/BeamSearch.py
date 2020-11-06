import numpy as np
from math import log
import tensorflow as tf
import time

class Video:
    def __init__(self,max_length,B):
        # max_length: total seconds in the video
        self.sequences=[[list(), 0.0,0,list()]]
        self.index_tracker={}
        self.max_length=max_length
        self.exclusive_beam_size=B

        # self.white_list = [] #to save the index of complete sequences in the beam

class BeamSearch:
    def __init__(self, z,b,l):
        self.zeta_=z
        self.beta_=b
        self.lambda_=l

    def batch_gen(self,xf, xa, xs,xa_v,xo_v,batch_size):  # data=[Total data points, T,F]   #label=[Total Data Points, 1]
        n = len(xf)
        batch_num = n // batch_size
        if batch_num==0:
            yield xf, xa, xs,xa_v,xo_v
        else:
            for b in range(batch_num):  # Here it generates batches of data within 1 epoch consecutively
                X_f = xf[batch_size * b:batch_size * (b + 1), :, :]
                X_a = xa[batch_size * b:batch_size * (b + 1)]
                X_s = xs[batch_size * b:batch_size * (b + 1)]
                X_a_v = xa_v[batch_size * b:batch_size * (b + 1), :]
                X_o_v = xo_v[batch_size * b:batch_size * (b + 1), :]
                yield X_f, X_a, X_s,X_a_v,X_o_v

            if n > batch_size * (b + 1):
                X_f = xf[batch_size * (b + 1):, :, :]
                X_a = xa[batch_size * (b + 1):]
                X_s = xs[batch_size * (b + 1):]
                X_a_v = xa_v[batch_size * (b+1):,:]
                X_o_v = xo_v[batch_size * (b + 1):, :]
                yield X_f, X_a, X_s,X_a_v,X_o_v

    def normalize(self,vec,k):
        n=len(vec)
        inds=np.argsort(vec)
        inds=inds[-k:]
        for i in range(n):
            if i not in inds:
                vec[i]=0.000001

        return vec
    def ignore_repetition(self,actions):
        actions=np.asarray(actions)
        temp = actions[1:] - actions[0:-1]
        idx = np.where(temp != 0)
        if len(idx) == 0:
            u_action = np.asarray(actions[-1])
            return u_action
        u_action=actions[idx]
        if len(actions!=0):
            u_action=np.append(u_action,actions[-1])

        return u_action
    def allignmentDONE(self,seqs,max_length,I):
        #seqs: a list of seqs
        #max_length: video's max_length in seconds
        #I: the first number of sequences that vote
        A2Vote={}
        MAX=-1
        best_seq_idx=-1
        if len(seqs)<I:
            return 0,0
        for i in range(I):
            seq, score, length_sofar, actions = seqs[i]
            if length_sofar == max_length:
                unique_actions=tuple(self.ignore_repetition(actions))
                if unique_actions not in A2Vote:
                    A2Vote[unique_actions]=1
                    if 1>MAX:
                        MAX=1
                        best_seq_idx=i
                else:
                    A2Vote[unique_actions] = A2Vote[unique_actions]+1
                    if A2Vote[unique_actions]>MAX:
                        MAX=A2Vote[unique_actions]
                        best_seq_idx=i
            else:
                return best_seq_idx,0
        assert MAX != -1, "MAX==-1"
        ref=list(self.ignore_repetition(seqs[best_seq_idx][3]))
        for j in range(I):
            if list(self.ignore_repetition(seqs[j][3]))==ref:
                best_seq_idx=j
                return best_seq_idx, 1


    def search(self,var_save_path,X_features,gt_actions,vid_info,vid_obj_info,VisModelTempPred,min_length_list,LengthNet,B,action2atomic_action,act2object_map,min_length_enable,mapact2act,action2step_map,action2bin_size):
        e=1e-20
        #X_features=[num_vid][(1/15)_sec_time_steps,feature_size] #time is in 1/15 sec scale
        # VisModelTempPred=[num_vid][1_sec_time_steps,num_actions]
        # gt_actions=[num_vid][#actions_per_this_video]
        #action2atomic_action: a dictionary to map the action to the verb (base action)
        vid_list=[Video(len(VisModelTempPred[v]),B) for v in range(len(X_features))]
        #read the pretrained length model
        self.sess = tf.Session(graph=LengthNet.graph)
        with LengthNet.graph.as_default():
            saver = tf.train.Saver()
        saver.restore(self.sess, var_save_path + 'my_model')
        ###################
        done_vid=0  #to track the number of alligned videos
        loop=0
        start=time.time()
        while(1):
            loop=loop+1
            xf = []
            xa = []
            xs=[]
            xa_v=[]
            xo_v=[]


            for v,video in enumerate(vid_list):

                for i in range(len(video.sequences)):
                    seq,_,length_sofar,actions = video.sequences[i]
                    if length_sofar == 0:

                        xa_v.append(vid_info[v,:])
                        xo_v.append(vid_obj_info[v,:])
                        xs.append(0)
                        xa.append(action2atomic_action[gt_actions[v][0]])
                        video.index_tracker[i]=1
                        if length_sofar  + LengthNet.duration > len(X_features[v]):
                            x = X_features[v][length_sofar :: LengthNet.step, :]
                            xf.append(np.concatenate((x, np.zeros([LengthNet.max_length - len(x), LengthNet.feature_size])),axis=0))
                        else:
                            xf.append(X_features[v][length_sofar :length_sofar + LengthNet.duration:LengthNet.step,:])
                        continue
                    if length_sofar == video.max_length and len(self.ignore_repetition(actions))==len(gt_actions[v]):
                        video.index_tracker[i] = 0
                        continue
                    if length_sofar == video.max_length and len(self.ignore_repetition(actions)) != len(gt_actions[v]):
                        video.index_tracker[i] = 0



                    prev_acts = self.ignore_repetition(actions)
                    if len(prev_acts)<=1:
                        xa_v.append(vid_info[v,:])
                        xo_v.append(vid_obj_info[v, :])
                    else:

                            xa_v.append(vid_info[v,:])
                            xo_v.append(vid_obj_info[v, :])



                    xa.append(action2atomic_action[actions[-1]])
                    if length_sofar + LengthNet.duration > len(X_features[v]):
                        x = X_features[v][length_sofar :: LengthNet.step, :]
                        xf.append(np.concatenate((x, np.zeros([LengthNet.max_length - len(x), LengthNet.feature_size])),axis=0))
                    else:
                        xf.append(X_features[v][length_sofar :length_sofar + LengthNet.duration:LengthNet.step, :])
                    ############
                    assert len(actions)!=0
                    last_action_idx = len(self.ignore_repetition(actions)) - 1

                    act_arr=np.asarray(actions)
                    temp=act_arr[1:]-act_arr[:-1]
                    ind = np.where(temp != 0)
                    if len(ind[0])!=0 :
                        lst_ind=ind[0][-1]+1
                        sec_behind = np.sum(seq[lst_ind:])
                        step=action2bin_size[action2atomic_action[actions[-1]]]
                        if sec_behind > step*LengthNet.length:
                            sec_behind = step*LengthNet.length
                        if sec_behind == 0:
                            sec_behind = 1
                        nSeg = (sec_behind - 1) // step

                    else:
                        sec_behind=np.sum(seq[:])
                        step = action2bin_size[action2atomic_action[actions[-1]]]
                        if sec_behind > step * LengthNet.length:
                            sec_behind = step * LengthNet.length
                        if sec_behind == 0:
                            sec_behind = 1
                        nSeg = (sec_behind - 1) // step

                    assert nSeg>-1 and nSeg<LengthNet.length

                    xs.append(nSeg)
                    ############
                    if last_action_idx == len(gt_actions[v]) - 1:
                        video.index_tracker[i]=1
                    else:
                        xs.append(0)

                        prev_acts = self.ignore_repetition(actions)
                        if len(prev_acts) <= 1:
                            xa_v.append(vid_info[v,:])
                            xo_v.append(vid_obj_info[v, :])
                        else:

                            xa_v.append(vid_info[v,:])
                            xo_v.append(vid_obj_info[v, :])
                        xa.append(action2atomic_action[gt_actions[v][last_action_idx + 1]])
                        video.index_tracker[i]=2

                        if length_sofar  + LengthNet.duration > len(X_features[v]):
                            x = X_features[v][length_sofar :: LengthNet.step, :]
                            xf.append(np.concatenate((x, np.zeros([LengthNet.max_length - len(x), LengthNet.feature_size])),axis=0))
                        else:
                            xf.append(X_features[v][length_sofar :length_sofar + LengthNet.duration:LengthNet.step,:])

            #check if all videos have been aligned
            if len(xa)==0 :#
                end=time.time()
                # print((end-start)*1000)
                self.sess.close()
                print(np.max(number_of_N))
                return [(video.sequences[0][0], video.sequences[0][3]) for video in vid_list]

            xa_v=np.asarray(xa_v)
            xo_v=np.asarray(xo_v)
            xf=np.asarray(xf)
            xa = np.asarray(xa)
            xs = np.asarray(xs)
            batch_size=50000
            for n_ep, data in enumerate(self.batch_gen(xf, xa, xs,xa_v,xo_v,batch_size)):
                _xf, _xa, _xs,_xa_v,_xo_v=data
                if n_ep==0:
                    with LengthNet.graph.as_default():
                        length_dist,obj_prob,base_prob = LengthNet.predict(self.sess, _xf, _xa,_xs,_xa_v,_xo_v)
                else:
                    with LengthNet.graph.as_default():
                        D,R,Ba = LengthNet.predict(self.sess, _xf, _xa, _xs,_xa_v,_xo_v)
                    length_dist=np.concatenate((length_dist,D),axis=0)
                    obj_prob = np.concatenate((obj_prob, R), axis=0)
                    base_prob = np.concatenate((base_prob, Ba), axis=0)

            number_of_N=np.zeros((len(vid_list)))
            assert len(length_dist)==len(xs)
            data_pointer=0
            for v,video in enumerate(vid_list):
                    all_candidates = list()
                    # expand each current candidate from our beam
                    for i in range(len(video.sequences)):
                        seq, score,length_sofar,actions = video.sequences[i]
                        assert length_sofar<=video.max_length,"length sofar is bigger than the length allowed"
                        if length_sofar==video.max_length and len(self.ignore_repetition(actions))==len(gt_actions[v]):
                            all_candidates.append(video.sequences[i])

                            continue

                        if length_sofar == video.max_length and len(self.ignore_repetition(actions)) != len(gt_actions[v]): #conflict with gt
                            continue

                        last_action_idx = len(self.ignore_repetition(actions)) - 1
                        ###################
                        if last_action_idx==len(gt_actions[v])-1 or last_action_idx==-1: # last action OR at the beginning when no action is picked
                            if last_action_idx==-1:
                                action_candidates = [gt_actions[v][0]]
                            else:
                                action_candidates=[gt_actions[v][last_action_idx]]
                        else:
                            action_candidates = [gt_actions[v][last_action_idx],gt_actions[v][last_action_idx+1]]

                        n_left_acts=len(gt_actions[v][last_action_idx + 1:])
                        if min_length_enable==True:
                            if n_left_acts > 2 and video.max_length-length_sofar<(n_left_acts*min(video.max_length//(len(gt_actions[v])+1),max(min_length_list[v],3*15))): # we estimate the left time is not enough to add other actions so dump this sequence
                                data_pointer = data_pointer + video.index_tracker[i]
                                continue
                        ############Making sure the sum of the candidate actions equal to 1 ##############
                        acts_prob = {}
                        t = e
                        for a in gt_actions[v]:  #1,5,30
                            acts_prob[a] = (VisModelTempPred[v][length_sofar][mapact2act[a]] ** self.lambda_) * (obj_prob[data_pointer, act2object_map[a]]**self.zeta_) * ((base_prob[data_pointer, action2atomic_action[a]]) ** self.beta_)
                        for a in acts_prob:
                            t = acts_prob[a] + t
                        for a in acts_prob:
                            acts_prob[a] = acts_prob[a] / t
                        ##################################################################################
                        for a in range(video.index_tracker[i]):
                            act_prob = acts_prob[action_candidates[a]]
                            assert xa[data_pointer]==action2atomic_action[action_candidates[a]],"Something wrong with stacking actions in order "+str(v)+" "+str(i)
                            assert video.index_tracker[i]==len(action_candidates),"something wrong with index tracking"
                            length_dist[data_pointer, :] = self.normalize(length_dist[data_pointer, :],3)
                            for j in range(len(length_dist[0,:])): #iterate over all length values
                                if length_dist[data_pointer, j] == 0:
                                    continue
                                    length_dist[data_pointer, j] = e
                                proposed_length=action2step_map[action2atomic_action[action_candidates[a]]]*j+min(15*3,action2step_map[action2atomic_action[action_candidates[a]]])
                                length_sofar_temp=length_sofar+proposed_length
                                if length_sofar_temp >= video.max_length:
                                    #cut off the rest
                                    length_sofar_temp=video.max_length
                                    overhead=length_sofar+proposed_length-video.max_length
                                    if overhead<=4*15:
                                        candidate = [seq + [proposed_length-overhead], score-log(length_dist[data_pointer,j])-log(act_prob+e), length_sofar_temp,actions+[action_candidates[a]]]
                                    else:
                                        break
                                else:
                                    candidate = [seq + [proposed_length], score-log(length_dist[data_pointer,j])-log(act_prob+e),length_sofar_temp,actions+[action_candidates[a]]]



                                #####
                                if length_sofar_temp != video.max_length or len( self.ignore_repetition(actions + [action_candidates[a]])) == len(gt_actions[v]):  # conflict with gt
                                    all_candidates.append(candidate)
                            data_pointer = data_pointer + 1  # points at the current sample in length_dist



                    if len(all_candidates)!=0:
                        ordered = sorted(all_candidates, key=lambda tup: tup[1]) # order all candidates by score
                        # select B best
                        bsize=video.exclusive_beam_size
                        video.sequences = ordered[:bsize]
                    else: #this happens when actions delay changing so we dont have any valid action seq in our beam, so we increase the beam size as well as the least expected duration for an action
                        print("all sequences in the beam have been pruned out, becasue they dont match the gt sequence,for video: " + str(v))
                        video.sequences = [[list(), 0.0,0,list()]]
                        video.exclusive_beam_size=video.exclusive_beam_size+100
                        if video.exclusive_beam_size>1200 and video.exclusive_beam_size<3000:
                            video.exclusive_beam_size=3000
                        if video.exclusive_beam_size > 3000 and video.exclusive_beam_size < 6000:
                            video.exclusive_beam_size = 6000
                        min_length_list[v]=min_length_list[v]+15
                        print(video.exclusive_beam_size)
                    nCompleted=0
                    #####Check if the first sequence for this video is full to throw out the rest
                    seq, score, length_sofar, actions = video.sequences[0]
                    if length_sofar == video.max_length and len(self.ignore_repetition(actions)) == len(gt_actions[v]):
                        nCompleted=nCompleted+1
                    if len(video.sequences)>1 and nCompleted==1:
                        done_vid=done_vid+1
                        video.sequences=[video.sequences[0]]
                        number_of_N[v]=len(video.sequences[0][0])
                        print("Number of aligned videos "+ str(done_vid)+ ' / '+ str(len(X_features)))
                        if done_vid==1430:
                            assert 1==1
            assert data_pointer==len(length_dist),"problem with data_pointer"






