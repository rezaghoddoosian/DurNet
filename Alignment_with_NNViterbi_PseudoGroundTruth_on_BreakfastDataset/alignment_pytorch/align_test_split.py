#!/usr/bin/python2.7

import numpy as np
from utils.dataset import Dataset
from utils.network_align import Trainer, Forwarder
from utils.viterbi import Viterbi
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', default=4, type=int)
parser.add_argument('--stage_id', default=1, type=int)
import sys
import pickle
#The code is obtained from https://github.com/alexanderrichard/NeuralNetwork-Viterbi and is modified accordingly.
def main(args):

    ### read label2index mapping and index2label mapping ###########################
    label2index = dict()
    index2label = dict()
    with open('data/mapping.txt', 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    ### read training data #########################################################
    print('read data...')
    with open('data/split{}.test'.format(str(args.split_id)), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    dataset = Dataset('data', video_list, label2index, shuffle = True)
    print('done')

    ### generate path grammar for inference ########################################
    paths = set()
    for sequence, transcript, videoname in dataset:
        paths.add( ' '.join([index2label[index] for index in transcript]))
    with open('align_results/grammar_split_{}.txt'.format(str(args.split_id)), 'w') as f:
        f.write('\n'.join(paths) + '\n')

    ### actual nn-viterbi training #################################################
    load_iteration = 10000
    if args.stage_id == 1:
        load_iteration = 10000
    else:
        split_iters_all = [11458, 11259, 11278, 11137]
        load_iteration = split_iters_all[args.split_id-1]

    length_model = PoissonModel('align_results/split_{}/stage_{}/lengths.iter-'.format(str(args.split_id), str(args.stage_id)) + str(load_iteration) + '.txt', max_length=2000)
    decoder = Viterbi(None, length_model, frame_sampling = 30, max_hypotheses = np.inf) # (None, None): transcript-grammar and length-model are set for each training sequence separately, see trainer.train(...)
    trainer = Trainer(decoder, dataset.input_dimension, dataset.n_classes, buffer_size = len(dataset), buffered_frame_ratio = 25)
    learning_rate = 0.01
    softmax_vals_path = 'align_results/split_{}/stage_{}/inference_val_split/softmax_values_align_test'.format(str(args.split_id), str(args.stage_id))
    with open(softmax_vals_path, 'rb') as in_f:
        softmax_vals = pickle.load(in_f)

    # train for 10000 iterations
    for i in range(10000):
        sequence, transcript, videoname = dataset.get()
        loss = trainer.train(sequence, transcript, videoname, load_iteration, softmax_vals, args.stage_id, args.split_id, batch_size = 1024, learning_rate = learning_rate)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)