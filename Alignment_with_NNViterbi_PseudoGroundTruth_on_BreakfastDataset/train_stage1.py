

import numpy as np
from utils.dataset import Dataset
from utils.network import Trainer, Forwarder
from utils.viterbi import Viterbi
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', default=4, type=int)
import sys
#The code is obtained from https://github.com/alexanderrichard/NeuralNetwork-Viterbi and is modified accordingly.
def main(args):
    # orig_stdout = sys.stdout
    # out_file = 'align_results/split_{}/out1.txt'.format(str(args.split_id))
    # # print(out_file)
    # f = open(out_file, 'w')
    # sys.stdout = f

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
    with open('data/split{}.train'.format(str(args.split_id)), 'r') as f:
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
    decoder = Viterbi(None, None, frame_sampling = 30, max_hypotheses = np.inf) # (None, None): transcript-grammar and length-model are set for each training sequence separately, see trainer.train(...)
    trainer = Trainer(decoder, dataset.input_dimension, dataset.n_classes, buffer_size = len(dataset), buffered_frame_ratio = 25)
    learning_rate = 0.01

    # train for 10000 iterations
    for i in range(10000):
        sequence, transcript, videoname = dataset.get()
        new_pseudo_gt = np.asarray([args.split_id])
        loss = trainer.train(sequence, transcript, videoname, new_pseudo_gt, batch_size = 1024, learning_rate = learning_rate)
        # print some progress information
        if (i+1) % 100 == 0:
            print('Iteration %d, loss: %f' % (i+1, loss))
        # save model every 1000 iterations
        if (i+1) % 1000 == 0:
            print('save snapshot ' + str(i+1))
            network_file = 'align_results/split_{}/stage_1/network.iter-'.format(str(args.split_id)) + str(i+1) + '.net'
            length_file = 'align_results/split_{}/stage_1/lengths.iter-'.format(str(args.split_id)) + str(i+1) + '.txt'
            prior_file = 'align_results/split_{}/stage_1/prior.iter-'.format(str(args.split_id)) + str(i+1) + '.txt'
            trainer.save_model(network_file, length_file, prior_file)
        # adjust learning rate after 2500 iterations
        if (i+1) == 2500:
            learning_rate = learning_rate * 0.1

    # sys.stdout = orig_stdout
    # f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
