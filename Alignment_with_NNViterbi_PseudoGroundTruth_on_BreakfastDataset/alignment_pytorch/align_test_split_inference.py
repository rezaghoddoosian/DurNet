#!/usr/bin/python2.7

import numpy as np
import multiprocessing as mp
import queue
from utils.dataset import Dataset
from utils.network_align import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi
import pickle

import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', default=3, type=int)
parser.add_argument('--stage_id', default=2, type=int)
import sys
#The code is obtained from https://github.com/alexanderrichard/NeuralNetwork-Viterbi and is modified accordingly.
def main(args):

    ### helper function for parallelized Viterbi decoding ##########################
    def decode(queue, log_probs, decoder, index2label):
        while not queue.empty():
            try:
                video = queue.get(timeout = 3)
                score, labels, segments = decoder.decode(log_probs[video])
                # save result
                with open('align_results/split_{}/stage_{}/viterbi_outputs_align_test_split/'.format(str(args.split_id), str(args.stage_id)) + video, 'w') as f:
                    # f.write( '### Recognized sequence: ###\n' )
                    # f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                    # f.write( '### Score: ###\n' + str(score) + '\n')
                    # f.write( '### Frame level recognition: ###\n')
                    f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )
            except queue.Empty:
                pass


    ### read label2index mapping and index2label mapping ###########################
    label2index = dict()
    index2label = dict()
    with open('data/mapping.txt', 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    ### read test data #############################################################
    with open('data/split{}.test'.format(args.split_id), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    dataset = Dataset('data', video_list, label2index, shuffle = False)

    # load prior, length model, grammar, and network
    load_iteration = 10000
    if args.stage_id == 1:
        load_iteration = 10000
    else:
        split_iters_all = [11458, 11259, 11278, 11137]
        load_iteration = split_iters_all[args.split_id-1]

    log_prior = np.log(np.loadtxt('align_results/split_{}/stage_{}/prior.iter-'.format(str(args.split_id), str(args.stage_id)) + str(load_iteration) + '.txt'))
    grammar = PathGrammar('align_results/grammar_split_{}.txt'.format(str(args.split_id)), label2index)
    length_model = PoissonModel('align_results/split_{}/stage_{}/lengths.iter-'.format(str(args.split_id), str(args.stage_id)) + str(load_iteration) + '.txt', max_length = 2000)
    forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
    forwarder.load_model('align_results/split_{}/stage_{}/network.iter-'.format(str(args.split_id), str(args.stage_id)) + str(load_iteration) + '.net')

    # parallelization
    n_threads = 8

    # Viterbi decoder
    viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf)
    # forward each video
    log_probs = dict()
    log_probs_smax = dict()
    queue = mp.Queue()
    # softmax_vals = []
    softmax_vals = {'test': 7}
    for i, data in enumerate(dataset):
        sequence, transcript, videoname = data
        print(i, videoname)
        video = list(dataset.features.keys())[i]
        queue.put(video)
        log_probs[video], log_probs_smax[video] = forwarder.forward(sequence)
        log_probs[video] -= log_prior #baye's
        log_probs[video] = log_probs[video] - np.max(log_probs[video])
        # row = [video, log_probs_smax[video], log_probs[video]]
        # softmax_vals.append(row)
        softmax_vals.update({video: [log_probs_smax[video], log_probs[video]]})

    file = open('align_results/split_{}/stage_{}/inference_val_split/softmax_values_align_test'.format(str(args.split_id), str(args.stage_id)), 'wb')
    pickle.dump(softmax_vals, file)
    file.close()

    # Viterbi decoding
    # procs = []
    # for i in range(n_threads):
    #     p = mp.Process(target = decode, args = (queue, log_probs, viterbi_decoder, index2label) )
    #     procs.append(p)
    #     p.start()
    # for p in procs:
    #     p.join()
    # sys.stdout = orig_stdout
    # f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

