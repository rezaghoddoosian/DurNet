

import numpy as np
import multiprocessing as mp
import queue
from utils.dataset import Dataset
from utils.network import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', default=4, type=int)
import sys

def main(args):


    ### read label2index mapping and index2label mapping ###########################
    label2index = dict()
    index2label = dict()
    with open('data/mapping.txt', 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    ### read test data #############################################################
    with open('data/split{}.test'.format(str(args.split_id)), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    dataset = Dataset('data', video_list, label2index, shuffle = False)

    # load prior, length model, grammar, and network
    split_iters_all =[11458, 11259, 11278, 11137]
    # load_iteration = 10000
    # load_iteration = 1458#split 2
    load_iteration = split_iters_all[args.split_id-1]
    log_prior = np.log(np.loadtxt('align_results/split_{}/stage_2/prior.iter-'.format(str(args.split_id)) + str(load_iteration) + '.txt') )
    grammar = PathGrammar('align_results/grammar_split_{}.txt'.format(args.split_id), label2index)
    length_model = PoissonModel('align_results/split_{}/stage_2/lengths.iter-'.format(str(args.split_id)) + str(load_iteration) + '.txt', max_length = 2000)
    forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
    forwarder.load_model('align_results/split_{}/stage_2/network.iter-'.format(str(args.split_id)) + str(load_iteration) + '.net')

    # parallelization
    n_threads = 8

    # Viterbi decoder
    viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf)
    # forward each video
    log_probs = dict()
    log_probs_smax = dict()
    queue = mp.Queue()
    softmax_vals = []
    for i, data in enumerate(dataset):
        sequence, transcript, videoname = data
        print(i, videoname)
        video = list(dataset.features.keys())[i]
        queue.put(video)
        log_probs[video], log_probs_smax[video] = forwarder.forward(sequence)
        log_probs[video] -= log_prior #baye's
        log_probs[video] = log_probs[video] - np.max(log_probs[video])
        row = [video, log_probs_smax[video]]
        softmax_vals.append(row)

    file = open('align_results/split_{}/stage_2/softmax_vals_test'.format(str(args.split_id)), 'wb')
    pickle.dump(softmax_vals, file)
    file.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
