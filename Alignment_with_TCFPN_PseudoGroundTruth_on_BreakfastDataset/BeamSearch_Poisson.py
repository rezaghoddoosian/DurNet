import numpy as np
from math import log
import math
import tensorflow as tf
class Video:
    def __init__(self,max_length,B):
        # max_length: total seconds in the video
        self.sequences=[[list(), 0.0,0,list()]]
        self.index_tracker={}
        self.max_length=max_length
        self.exclusive_beam_size=B

        # self.white_list = [] #to save the index of complete sequences in the beam

class BeamSearch_Poisson:
    def __init__(self,zeta_,beta_,lambda_):
        self.zeta_=zeta_
        self.beta_=beta_
        self.lambda_=lambda_

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

    def poisson(self,gamma,length):
        Factorial=0
        for i in range(1,length+1):
            Factorial=np.log(i)+Factorial

        logP=length*np.log(gamma)-gamma-Factorial
        return logP

    def search(self,var_save_path,X_features,gt_actions,vid_info,vid_obj_info,VisModelTempPred,min_length_list,LengthNet,B,length_cap,action2atomic_action,act2object_map,min_length_enable,action2step_map,action2bin_size,gamma):
        new_obj_prob = np.load("object_prob.npy")
        e=1e-20
        #X_features=[num_vid][(1/15)_sec_time_steps,feature_size] #time is in 1/15 sec scale
        # VisModelTempPred=[num_vid][1_sec_time_steps,num_actions]
        # gt_actions=[num_vid][#actions_per_this_video]
        #length_cap : in seconds indicates the maximum length allowed for each action
        #action2atomic_action: a dictionary to map the action to the verb (base action)
        vid_list=[Video(len(VisModelTempPred[v]),B) for v in range(len(X_features))]
        #read the pretrained length model
        # saver = tf.train.import_meta_graph('my_model.meta')

        # saver = tf.train.Saver()
        # self.sess = tf.Session(graph=LengthNet.graph)
        # saver.restore(self.sess, var_save_path + 'my_model')

        # read the pretrained length model
        self.sess = tf.Session(graph=LengthNet.graph)
        with LengthNet.graph.as_default():
            saver = tf.train.Saver()
        saver.restore(self.sess, var_save_path + 'my_model')
        ###################
        done_vid=0  #to track the number of alligned videos
        loop=0
        while(1):
            loop=loop+1
            xf = []
            xa = []
            xs=[]
            xa_v=[]
            xo_v=[]
            count=0
            C=0
            for v,video in enumerate(vid_list):
                if loop==1:
                    for l,ac in enumerate(gt_actions[v]):
                        if l==0:
                            continue
                        if action2atomic_action[gt_actions[v][l]]==action2atomic_action[gt_actions[v][l-1]]:
                            count=count+1
                        C=C+1

                # to find out if, in any video, the seqs in the beam conflict with the gt sequence so to increment the beam size and redo the search
                for i in range(len(video.sequences)):
                    seq,_,length_sofar,actions = video.sequences[i]



                    if length_sofar == 0:
                        input = np.zeros([LengthNet.nClasses])
                        xa_v.append(vid_info[v,:])
                        xo_v.append(vid_obj_info[v,:])
                        xs.append(0)
                        xa.append(action2atomic_action[gt_actions[v][0]])
                        video.index_tracker[i]=1
                        if length_sofar * 15 + LengthNet.duration > len(X_features[v]):
                            x = X_features[v][length_sofar * 15:: LengthNet.step, :]
                            xf.append(
                                np.concatenate((x, np.zeros([LengthNet.max_length - len(x), LengthNet.feature_size])),
                                               axis=0))
                        else:
                            xf.append(X_features[v][length_sofar * 15:length_sofar * 15 + LengthNet.duration:LengthNet.step,:])
                        continue
                    if length_sofar == video.max_length and len(self.ignore_repetition(actions))==len(gt_actions[v]):
                        counter2 = 0
                        # video.white_list.append(i)
                        video.index_tracker[i] = 0
                        continue
                    if length_sofar == video.max_length and len(self.ignore_repetition(actions)) != len(gt_actions[v]):
                        assert 1==2
                        video.index_tracker[i] = 0
                        counter2=counter2+1
                        if counter2==len(video.sequences):   #increment the beam size and redo the search
                            print("increment the beam size {}".format(B+2))
                            return self.search(var_save_path, X_features, gt_actions, VisModelTempPred, LengthNet, B+2)
                        continue

                    input = np.zeros([LengthNet.nClasses])
                    prev_acts = self.ignore_repetition(actions)
                    if len(prev_acts)<=1:
                        xa_v.append(vid_info[v,:])
                        xo_v.append(vid_obj_info[v, :])
                    else:
                            # input = LengthNet.create_label(input, action2atomic_action, prev_acts)
                            xa_v.append(vid_info[v,:])
                            xo_v.append(vid_obj_info[v, :])



                    xa.append(action2atomic_action[actions[-1]])
                    if length_sofar * 15 + LengthNet.duration > len(X_features[v]):
                        x = X_features[v][length_sofar * 15:: LengthNet.step, :]
                        xf.append(np.concatenate((x, np.zeros([LengthNet.max_length - len(x), LengthNet.feature_size])),axis=0))
                    else:
                        xf.append(X_features[v][length_sofar * 15:length_sofar * 15 + LengthNet.duration:LengthNet.step, :])
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
                    #######    to avoid very long sequences actions
                    if sec_behind>length_cap[actions[-1]]+3000 and last_action_idx!=len(gt_actions[v])-1:
                        video.index_tracker[i] = 1
                        xa[-1]=action2atomic_action[gt_actions[v][last_action_idx + 1]]
                        xs.append(0)
                        continue
                    #######
                    if nSeg>15:
                        nSeg=15
                    xs.append(nSeg)

                    ############
                    if last_action_idx == len(gt_actions[v]) - 1:
                        video.index_tracker[i]=1
                    else:
                        xs.append(0)
                        input = np.zeros([LengthNet.nClasses])
                        prev_acts = self.ignore_repetition(actions)
                        if len(prev_acts) <= 1:
                            xa_v.append(vid_info[v,:])
                            xo_v.append(vid_obj_info[v, :])
                        else:
                            # input = LengthNet.create_label(input, action2atomic_action, prev_acts)
                            xa_v.append(vid_info[v,:])
                            xo_v.append(vid_obj_info[v, :])
                        xa.append(action2atomic_action[gt_actions[v][last_action_idx + 1]])
                        video.index_tracker[i]=2

                        if length_sofar * 15 + LengthNet.duration > len(X_features[v]):
                            x = X_features[v][length_sofar * 15:: LengthNet.step, :]
                            xf.append(np.concatenate((x, np.zeros([LengthNet.max_length - len(x), LengthNet.feature_size])),axis=0))
                        else:
                            xf.append(X_features[v][length_sofar * 15:length_sofar * 15 + LengthNet.duration:LengthNet.step,:])

            # print(count, C)
            #check if all videos have been aligned
            if len(xa)==0 :#or (len(xa)>2000 and len(gt_actions)>1000)
                self.sess.close()
                return [(video.sequences[0][0], video.sequences[0][3]) for video in vid_list]

            xa_v=np.asarray(xa_v)
            xo_v=np.asarray(xo_v)
            xf=np.asarray(xf)
            xa = np.asarray(xa)
            xs = np.asarray(xs)
            batch_size=50000
            # length_dist=[bath_size,n_length]
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
                        flag=False
                        if len(actions) > 0:
                            act_arr = np.asarray(actions)
                            if len(actions)!=1:
                                temp = act_arr[1:] - act_arr[:-1]
                                ind = np.where(temp != 0)
                                if len(ind[0]) != 0:
                                    lst_ind = ind[0][-1] + 1
                                    sec_behind = np.sum(seq[lst_ind:])
                                else:
                                    sec_behind = np.sum(seq[:])
                            else:
                                sec_behind = np.sum(seq[:])
                        # if the current action has been happening for too long and is not the last action, we force it to change to the next one
                            if (sec_behind > length_cap[actions[-1]]+3000) and last_action_idx!=len(gt_actions[v])-1:
                                action_candidates=[gt_actions[v][last_action_idx + 1]]
                                flag=True
                        ###################
                        if flag == False:
                            if last_action_idx==len(gt_actions[v])-1 or last_action_idx==-1: # last action OR at the beginning when no action is picked
                                if last_action_idx==-1:
                                    action_candidates = [gt_actions[v][0]]
                                else:
                                    action_candidates=[gt_actions[v][last_action_idx]]
                            else:
                                action_candidates = [gt_actions[v][last_action_idx],gt_actions[v][last_action_idx+1]]

                        n_left_acts=len(gt_actions[v][last_action_idx + 1:])
                        if min_length_enable==True:
                            if n_left_acts > 2 and video.max_length-length_sofar<(n_left_acts*min(video.max_length//(len(gt_actions[v])+1),max(min_length_list[v],3))): # we estimate the left time is not enough to add other actions so dump this sequence
                                data_pointer = data_pointer + video.index_tracker[i]
                                continue
                        ############Making sure the sum of the candidate action probabilities equal to 1 ##############
                        acts_prob={}
                        t = e
                        for a in gt_actions[v]:
                            acts_prob[a] = (VisModelTempPred[v][length_sofar][a]**self.lambda_)*(obj_prob[data_pointer,act2object_map[a]]**self.zeta_) * ((base_prob[data_pointer, action2atomic_action[a]]) ** self.beta_)
                        for a in acts_prob:
                            t = acts_prob[a] + t
                        for a in acts_prob:
                            acts_prob[a] = acts_prob[a] / t
                        ##################################################################################
                        for a in range(video.index_tracker[i]):
                            act_prob = acts_prob[action_candidates[a]]
                            assert xa[data_pointer]==action2atomic_action[action_candidates[a]],"Something wrong with stacking actions in order "+str(v)+" "+str(i)
                            vis_model_prob = VisModelTempPred[v][length_sofar][action_candidates[a]]
                            object_prob=obj_prob[data_pointer,act2object_map[action_candidates[a]]]
                            assert video.index_tracker[i]==len(action_candidates),"something wrong with index tracking"+str(v)+","+str(i)

                            if v==27:
                                assert 1==1

                            for j in range(len(length_dist[0, :])):
                                proposed_length = action2step_map[action2atomic_action[action_candidates[a]]] * j + min(3, action2step_map[action2atomic_action[action_candidates[a]]])
                                # proposed_length=5*j+3
                                if len(actions) > 0:
                                    if action2atomic_action[action_candidates[a]]==action2atomic_action[actions[-1]]:
                                        length_dist[data_pointer, j] =self.poisson(gamma[action2atomic_action[action_candidates[a]]],sec_behind+proposed_length)
                                    else:
                                        length_dist[data_pointer, j] =self.poisson(gamma[action2atomic_action[action_candidates[a]]],proposed_length)
                                else:
                                    length_dist[data_pointer, j] = self.poisson(gamma[action2atomic_action[action_candidates[a]]], proposed_length)
                                    assert len(action_candidates)==1
                                length_sofar_temp=length_sofar+proposed_length
                                best_base_prob = base_prob[data_pointer, action2atomic_action[action_candidates[a]]]
                                # object_prob=new_obj_prob [v][length_sofar][act2object_map[action_candidates[a]]]
                                if length_sofar_temp >= video.max_length:
                                    #cut off the rest
                                    length_sofar_temp=video.max_length
                                    overhead=length_sofar+proposed_length-video.max_length
                                    if overhead<=4:
                                        candidate = [seq + [proposed_length-overhead], score-1*length_dist[data_pointer,j]-1*log(act_prob+e), length_sofar_temp,actions+[action_candidates[a]]]
                                    else:
                                        break
                                else:

                                    candidate = [seq + [proposed_length], score-1*length_dist[data_pointer,j]-1*log(act_prob+e),length_sofar_temp,actions+[action_candidates[a]]]
                                ###### if the current action has been happening for too long
                                ss=seq + [proposed_length]
                                aa=actions + [action_candidates[a]]
                                act_arr = np.asarray(aa)
                                if len(aa) != 1:
                                    temp = act_arr[1:] - act_arr[:-1]
                                    ind = np.where(temp != 0)
                                    if len(ind[0]) != 0:
                                        lst_ind = ind[0][-1] + 1
                                        sec_behind = np.sum(ss[lst_ind:])
                                    else:
                                        sec_behind = np.sum(ss[:])
                                else:
                                    sec_behind = np.sum(ss[:])

                                if (sec_behind > length_cap[aa[-1]]+20+3000) :
                                    continue
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
                        video.exclusive_beam_size=video.exclusive_beam_size+200
                        if video.exclusive_beam_size>1200 and video.exclusive_beam_size<3000:
                            video.exclusive_beam_size=3000
                        if video.exclusive_beam_size > 3000 and video.exclusive_beam_size < 6000:
                            video.exclusive_beam_size = 6000
                        min_length_list[v]=min_length_list[v]+1
                        print(video.exclusive_beam_size)
                    nCompleted=0
                    #####Check if the first sequence for this video is full to throw out the rest
                    seq, score, length_sofar, actions = video.sequences[0]
                    if length_sofar == video.max_length and len(self.ignore_repetition(actions)) == len(gt_actions[v]):
                        nCompleted=nCompleted+1
                    if len(video.sequences)>1 and nCompleted==1:
                        done_vid=done_vid+1
                        video.sequences=[video.sequences[0]]
                        print("Number of done videos "+ str(done_vid)+ ' / '+ str(len(X_features)))

            assert data_pointer==len(length_dist),"problem with data_pointer"






##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################











    def test_search(self,var_save_path,lang_var_save_path,X_features,vid_info,vid_obj_info,VisModelTempPred,LengthNet,LangNet,B,length_cap,action2atomic_action,alpha,per_sec_y_test,act2object_map,gt_actions):
        #var_save_path= path to load the encoder decoder
        #lang_var_save_path= path to load the language model
        #X_features=[num_vid][(1/15)_sec_time_steps,feature_size] #time is in 1/15 sec scale
        # VisModelTempPred=[num_vid][1_sec_time_steps,num_actions]
        #length_cap : in seconds indicates the maximum length allowed for each action
        #action2atomic_action: a dictionary to map the action to the verb (base action)
        #B= beam_size
        new_obj_prob = np.load("object_prob_test.npy")
        def lang_batch_gen(lang_xa,batch_size):
            n = len(lang_xa)
            batch_num = n // batch_size
            if batch_num == 0:

                yield lang_xa

            else:
                for b in range(batch_num):  # Here it generates batches of data within 1 epoch consecutively
                    X_langxa = lang_xa[batch_size * b:batch_size * (b + 1), :, :]
                    yield X_langxa

                if n > batch_size * (b + 1):
                    X_langxa = lang_xa[batch_size * (b + 1):, :, :]
                    yield X_langxa

        e = 1e-20
        vid_list=[Video(len(VisModelTempPred[v]),B) for v in range(len(X_features))]
        #read the pretrained length model
        self.sess = tf.Session(graph=LengthNet.graph)
        with LengthNet.graph.as_default():
            saver = tf.train.Saver()
        saver.restore(self.sess, var_save_path + 'my_model')
        ###################
        #read the pretrained language model
        with LangNet.graph.as_default():
            saver = tf.train.Saver()
        lang_sess = tf.Session(graph=LangNet.graph)
        saver.restore(lang_sess, lang_var_save_path + 'my_language_model')

        done_vid=0  #to track the number of alligned videos
        MIN_ACT_TO_START_LANG=1
        loop=0

        while(1):
            loop=loop+1
            xf = []
            xa = []
            xs=[]
            xa_v=[]
            xo_v = []
            first_langx_picked=False
            for v,video in enumerate(vid_list):

                # to find out if, in any video, the seqs in the beam conflict with the gt sequence so to increment the beam size and redo the search
                for i in range(len(video.sequences)):
                    seq,_,length_sofar,actions = video.sequences[i]

                    if length_sofar == video.max_length :
                        video.index_tracker[i] = 0
                        continue

                    video.index_tracker[i] = 0
                    if len(actions) != 0:
                        uniqe_actions=self.ignore_repetition(actions)
                        if len(uniqe_actions) > MIN_ACT_TO_START_LANG: #only if the context exists
                            if len(uniqe_actions) > LangNet.max_length:
                                print("unique actions exceeded than expected")
                                uniqe_actions=uniqe_actions[-LangNet.max_length:]
                            action_emb = LangNet.create_action_embedding(lang_sess, uniqe_actions)
                            if first_langx_picked == False:
                                lang_xa=action_emb
                                first_langx_picked=True
                            else:
                                lang_xa = tf.concat([lang_xa, action_emb], 0)

                            # lang_xa.append(action_emb)
                    ############################
                    # inp = np.zeros([LengthNet.nClasses])
                    inp=gt_actions[v]
                    if len(actions) != 0:
                        tmp = LengthNet.create_label(inp, action2atomic_action, uniqe_actions)
                    for ac in range(LengthNet.nClasses):
                        if len(actions) == 0 and ac!=0: #makes sure the actions start with bg
                            continue
                        video.index_tracker[i] = video.index_tracker[i] + 1
                        xa.append(ac)
                        if len(actions) != 0:
                            xo_v.append(vid_obj_info[v, :])
                            xa_v.append(vid_info[v, :])
                            # xa_v.append(inp)
                            # xa_v.append(LengthNet.create_label(inp, action2atomic_action, gt_actions[v]))
                            if action2atomic_action[actions[-1]] ==ac:  # if the next action is of the same type
                                act_arr=np.asarray(actions)
                                temp=act_arr[1:]-act_arr[:-1]
                                ind = np.where(temp != 0)
                                if len(ind[0])!=0:
                                    lst_ind=ind[0][-1]+1
                                    sec_behind = np.sum(seq[lst_ind:])
                                    nSeg=sec_behind//LengthNet.sample_rate

                                else:
                                    sec_behind=np.sum(seq[:])
                                    nSeg = sec_behind // LengthNet.sample_rate
                                if nSeg > 15:
                                    nSeg = 15
                                xs.append(nSeg)

                                if sec_behind > length_cap[actions[-1]]+25 or (len(uniqe_actions)==1 and ac==0):
                                    video.index_tracker[i] = video.index_tracker[i] - 1
                                    xa.pop()
                                    xa_v.pop()
                                    xo_v.pop()
                                    xs.pop()
                                    continue



                            else:
                                xs.append(0)
                        else:
                            xs.append(0)
                            xo_v.append(vid_obj_info[v, :])
                            xa_v.append(vid_info[v,:])
                            # xa_v.append(inp)
                            # xa_v.append(LengthNet.create_label(inp, action2atomic_action, gt_actions[v]))

                        if length_sofar * 15 + LengthNet.duration > len(X_features[v]):
                            x = X_features[v][length_sofar * 15:: LengthNet.step, :]
                            xf.append(np.concatenate((x, np.zeros([LengthNet.max_length - len(x), LengthNet.feature_size])), axis=0))

                        else:
                            xf.append(X_features[v][length_sofar * 15:length_sofar * 15 + LengthNet.duration:LengthNet.step,:])



            #check if all videos have been aligned
            if len(xa)==0:
                self.sess.close()
                lang_sess.close()
                return [(video.sequences[0][0], video.sequences[0][3]) for video in vid_list]
            print(loop,len(xa),vid_list[1].sequences[0][3])
            batch_size = 50000
            xa_v=np.asarray(xa_v)
            xo_v=np.asarray(xo_v)
            xf=np.asarray(xf)
            xa = np.asarray(xa)
            xs = np.asarray(xs)
            if first_langx_picked==True:
                num_seqs=int(lang_xa.get_shape().as_list()[0]/LangNet.max_length)
                assert lang_xa.get_shape().as_list()[0]%LangNet.max_length==0
                # for i in range(LangNet.max_length):
                #     lang_xa_tmp = lang_sess.run(lang_xa[i*num_seqs:(i+1)*num_seqs,:])
                #     if i==0:
                #         lang_xa_array=lang_xa_tmp
                #     else:
                #         lang_xa_array=np.concatenate((lang_xa_array,lang_xa_tmp), axis=0)
                lang_xa_array=lang_sess.run(lang_xa)
                print(len(lang_xa_array))
                # lang_xa_array=np.asarray(lang_xa_list)
                print("after!!")
                lang_xa=[]
                for i in range(num_seqs):
                    lang_xa.append(lang_xa_array[i*LangNet.max_length:(i+1)*LangNet.max_length,:])

                lang_xa = np.asarray(lang_xa)
                assert (i+1) * LangNet.max_length==len(lang_xa_array),"issue with partitioning lang_xa_array "

                for n_ep, data in enumerate(lang_batch_gen(lang_xa, batch_size)):
                    _lang_xa=data
                    if n_ep == 0:
                        lang_dist = LangNet.predict(lang_sess, _lang_xa)
                    else:
                        La = LangNet.predict(lang_sess, _lang_xa)
                        lang_dist = np.concatenate((lang_dist, La), axis=0)


            for n_ep, data in enumerate(self.batch_gen(xf, xa, xs,xa_v,xo_v,batch_size)):
                _xf, _xa, _xs, _xa_v,_xo_v=data
                if n_ep==0:# length_dist=[bath_size,n_length]
                    length_dist, obj_prob, base_prob = LengthNet.predict(self.sess, _xf, _xa,_xs,_xa_v,_xo_v)
                else:
                    D,R,Ba = LengthNet.predict(self.sess, _xf, _xa, _xs,_xa_v,_xo_v)
                    length_dist=np.concatenate((length_dist,D),axis=0)
                    obj_prob = np.concatenate((obj_prob, R), axis=0)
                    base_prob = np.concatenate((base_prob, Ba), axis=0)

            if first_langx_picked == True:
                assert len(lang_dist) == len(lang_xa),str(lang_dist)+" "+str(lang_xa)
            assert len(length_dist)==len(xs)

            ref=0
            lang_data_pointer=0
            for v,video in enumerate(vid_list):
                    if v == 6 :
                        assert 1 == 1
                    all_candidates = list()
                    # expand each current candidate from our beam
                    for i in range(len(video.sequences)):
                        seq, score,length_sofar,actions = video.sequences[i]
                        assert length_sofar<=video.max_length,"length sofar is bigger than the length allowed"
                        if length_sofar==video.max_length:
                            all_candidates.append(video.sequences[i])

                            continue
                        uniqe_actions = self.ignore_repetition(actions)
                        ###################
                        flag=False
                        action_candidates=[]
             ####
                        for ac in range(LengthNet.nActions):
                            if len(actions) == 0:
                                action_candidates.append(ac)
                                break #########&&&&&#########
                            else:
                                if action2atomic_action[actions[-1]] == action2atomic_action[ac]:
                                    act_arr = np.asarray(actions)
                                    action_candidates.append(ac)
                                    if len(actions) != 1:
                                        temp = act_arr[1:] - act_arr[:-1]
                                        ind = np.where(temp != 0)
                                        if len(ind[0]) != 0:
                                            lst_ind = ind[0][-1] + 1
                                            sec_behind = np.sum(seq[lst_ind:])
                                        else:
                                            sec_behind = np.sum(seq[:])
                                    else:
                                        sec_behind = np.sum(seq[:])
                                    if (sec_behind > length_cap[actions[-1]]+25 or (len(uniqe_actions)==1 and ac==0) ):
                                        action_candidates.pop()
                                        flag=True
                                else:
                                    action_candidates.append(ac)


                        if len(uniqe_actions) <= MIN_ACT_TO_START_LANG:
                            lang_model_prob = 1 - e
                            lang_data_pointer = lang_data_pointer - 1
                        for a in range(len(action_candidates)):
                            offset=action2atomic_action[action_candidates[a]]
                            if flag==True :# if the type of action must change
                                if offset>action2atomic_action[actions[-1]]:
                                    offset=offset-1
                            data_pointer=offset+ref
                            assert xa[data_pointer]==action2atomic_action[action_candidates[a]],"Something wrong with stacking actions in order "+str(v)+" "+str(i)
                            assert video.index_tracker[i]==LengthNet.nClasses or video.index_tracker[i]==LengthNet.nClasses-1 or video.index_tracker[i]==1,"video.index_tracker wrong!"+str(video.index_tracker[i])
                            if len(uniqe_actions)>MIN_ACT_TO_START_LANG:  # use the language model only when the previous actions create a context
                                lang_model_prob=lang_dist[lang_data_pointer,action_candidates[a]]

                            vis_model_prob = VisModelTempPred[v][length_sofar][action_candidates[a]]
                            object_prob = obj_prob[data_pointer, act2object_map[action_candidates[a]]]
                            # assert video.index_tracker[i]==len(action_candidates),"something wrong with index tracking"
                            length_dist[data_pointer, :] = self.normalize(length_dist[data_pointer, :],3)
                            for j in range(len(length_dist[0,:])): #iterate over all length values
                                if length_dist[data_pointer, j] == 0:
                                    length_dist[data_pointer, j] = 2e-20
                                proposed_length=5*j+3
                                best_base_prob=base_prob[data_pointer, action2atomic_action[action_candidates[a]]]
                                # object_prob = new_obj_prob[v][length_sofar][act2object_map[action_candidates[a]]]
                                length_sofar_temp=length_sofar+proposed_length
                                if length_sofar_temp >= video.max_length:
                                    #cut off the rest
                                    length_sofar_temp=video.max_length
                                    overhead=length_sofar+proposed_length-video.max_length
                                    if overhead<=4:
                                        candidate = [seq + [proposed_length-overhead], score-1*log(length_dist[data_pointer,j])-1*log(vis_model_prob+e)-1*log(lang_model_prob+e)-1*log(best_base_prob+e)-2*log(object_prob+e), length_sofar_temp,actions+[action_candidates[a]]]
                                    else:
                                        break
                                else:
                                    candidate = [seq + [proposed_length], score-1*log(length_dist[data_pointer,j])-1*log(vis_model_prob+e)-1*log(lang_model_prob+e)-1*log(best_base_prob+e)-2*log(object_prob+e),length_sofar_temp,actions+[action_candidates[a]]]
                                ###### if the current action has been happening for too long
                                ss=seq + [proposed_length]
                                aa=actions + [action_candidates[a]]
                                act_arr = np.asarray(aa)
                                if len(aa) != 1:
                                    temp = act_arr[1:] - act_arr[:-1]
                                    ind = np.where(temp != 0)
                                    if len(ind[0]) != 0:
                                        lst_ind = ind[0][-1] + 1
                                        sec_behind = np.sum(ss[lst_ind:])
                                    else:
                                        sec_behind = np.sum(ss[:])
                                else:
                                    sec_behind = np.sum(ss[:])

                                if (sec_behind > length_cap[aa[-1]]+20+25) :
                                    continue
                                #####
                                all_candidates.append(candidate)
                        ref = ref + video.index_tracker[i]  # points at the current sample in length_dist
                        lang_data_pointer=lang_data_pointer+1

                    if len(all_candidates)!=0:
                        ordered = sorted(all_candidates, key=lambda tup: tup[1]) # order all candidates by score
                        # select B best
                        bsize=video.exclusive_beam_size
                        video.sequences = ordered[:bsize]


                    #####Check if a valid allignment is found (valid means if it covers the whole video and its action set is repeated the most in the first I seqs of the beam)
                    best_idx,Completed=self.allignmentDONE(video.sequences, video.max_length, I=20)
                    if v<10 and Completed==1:
                        [print(video.sequences[i][3]) for i in range(20)]
                        print("vid "+str(v)+" Completed")
                    if len(video.sequences)>1 and Completed==1:
                        done_vid=done_vid+1
                        video.sequences=[video.sequences[best_idx]]
                        print("Number of done videos "+ str(done_vid)+ ' / '+ str(len(X_features)))
            assert ref==len(length_dist),"problem with data_pointer"
            if first_langx_picked == True:
                assert lang_data_pointer == len(lang_dist), "problem with language_data_pointer"
