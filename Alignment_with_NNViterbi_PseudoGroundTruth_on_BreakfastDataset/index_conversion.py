import pickle
import numpy as np


class index_conversion:






    def map_index(self,prev,map_list):
        new=np.zeros([len(prev)],dtype=int)
        for i in range(1,48):
            ind= prev==i
            if sum(ind)!=0:
                new[ind]=int(map_list[i])

        return new

    def NNViterbi_to_Ours(self):
        for split in [1,2,3,4]:

            annot_fname="./align_results/split_{}/stage_1/nn_viterbi_stage_1_vals_train".format(split)
            with open(annot_fname, 'rb') as in_f:
                annot_data = pickle.load(in_f)


            predicted_weak=[]
            predicted_val=[]
            maplist=[]
            actions = open("act2actmap.txt").readlines()
            for i in range(48):
                maplist.append(actions[i].split()[1])
            for v in range(len(annot_data)):
                print(v)
                predicted_val.append(np.asarray(annot_data[v][2])[4::])
                predicted_weak.append(annot_data[v][1][4:,:])
                predicted_val[-1]=self.map_index(predicted_val[-1],maplist)
            np.save("weak_predictions{}.npy".format(split),predicted_weak)
            np.save("predictions{}.npy".format(split),predicted_val)

    def Ours_to_NNViterbi(self):
        for split in [1, 2, 3, 4]:
            annot_fname = "./align_results/split_{}/stage_1/nn_viterbi_stage_1_vals_train".format(split)
            with open(annot_fname, 'rb') as in_f:
                annot_data = pickle.load(in_f)

            pseudo_gt_list=[]
            ps=np.load("pseudo_gt_{}.npy".format(split))
            mapback_dic={}
            actions = open("act2actmap.txt").readlines()
            for i in range(48):  #mapping my labels to baseline labels
                mapback_dic[int(actions[i].split()[1])]=int(actions[i].split()[0])
            ps_new=[]
            for v in range(len(ps)):
                print(v)
                a=self.map_index(ps[v], mapback_dic)
                a=np.concatenate((a,a[-1]*np.ones((4),dtype=int)))
                ps_new.append(a)
            for i in range(len(ps)):
                print(i)
                pseudo_gt_list.append([])
                pseudo_gt_list[-1].append(annot_data[i][0])
                assert len(annot_data[i][1])==len(ps_new[i])
                pseudo_gt_list[-1].append(ps_new[i])
            assert len(pseudo_gt_list)==len(annot_data)
            np.save("./align_results/new_pseudo_gt_split{}_train.npy".format(split),pseudo_gt_list)


    def create_GRU_test(self):
        for split in [1, 2, 3, 4]:
            annot_fname = "./align_results/split_{}/stage_2/softmax_vals_test".format(split)
            with open(annot_fname, 'rb') as in_f:
                annot_data = pickle.load(in_f)

            predicted_weak = []
            for v in range(len(annot_data)):
                print(v)
                predicted_weak.append(annot_data[v][1][4:, :])

            np.save("GRU_split{}_test.npy".format(split), predicted_weak)










