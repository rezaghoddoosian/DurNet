
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
import csv
#The code is obtained from https://github.com/alexanderrichard/NeuralNetwork-Viterbi and is modified accordingly.
def main(args):
    ### helper function for parallelized Viterbi decoding ##########################
    def decode(queue, log_probs, decoder, index2label):
        while not queue.empty():
            try:
                video = queue.get(timeout = 3)
                score, labels, segments = decoder.decode(log_probs[video])
                # save result
                with open('align_results/split_{}/stage_1/inference_train_split/'.format(str(args.split_id)) + video, 'w') as f:
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
    with open('data/split{}.train'.format(str(args.split_id)), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    dataset = Dataset('data', video_list, label2index, shuffle = False)

    # load prior, length model, grammar, and network
    load_iteration = 10000
    # load_iteration = 1458
    log_prior = np.log(np.loadtxt('align_results/split_{}/stage_1/prior.iter-'.format(str(args.split_id)) + str(load_iteration) + '.txt') )
    grammar = PathGrammar('align_results/grammar_split_{}.txt'.format(str(args.split_id)), label2index)
    length_model = PoissonModel('align_results/split_{}/stage_1/lengths.iter-'.format(str(args.split_id)) + str(load_iteration) + '.txt', max_length = 2000)
    forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
    forwarder.load_model('align_results/split_{}/stage_1/network.iter-'.format(str(args.split_id)) + str(load_iteration) + '.net')

    # parallelization
    n_threads = 8
    print('data and model loading done!!!')


    # Viterbi decoder
    viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf)
    # forward each video
    log_probs = dict()
    log_probs_smax = dict()
    queue = mp.Queue()
    softmax_vals = []
    for i, data in enumerate(dataset):
        sequence, transcript, videoname = data
        video = list(dataset.features.keys())[i]
        queue.put(video)
        log_probs[video], log_probs_smax[video] = forwarder.forward(sequence)
        log_probs[video] -= log_prior #baye's
        log_probs[video] = log_probs[video] - np.max(log_probs[video])
        row = [video, log_probs_smax[video]]
        softmax_vals.append(row)

    print('Softmax GRU done!')

    file = open('align_results/split_{}/stage_1/softmax_vals_train'.format(str(args.split_id)), 'wb')
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


    ##############Consolidate the results
    fname = 'align_results/split_{}/stage_1/softmax_vals_train'.format(str(args.split_id))
    with open(fname, 'rb') as in_f:
        data = pickle.load(in_f)

    label2index = dict()
    index2label = dict()
    with open('data/mapping.txt', 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    final_out = []
    result_folder = 'align_results/split_{}/stage_1/viterbi_outputs_training/'.format(str(args.split_id))

    for iter in range(len(data)):
        sample_name = data[iter][0]
        filename = result_folder + sample_name
        row = []
        with open(result_folder + sample_name, 'rb') as in_f:
            preds = pickle.load(in_f)
            row = [data[iter][0], data[iter][1], np.asarray(preds)]

        # with open(result_folder + sample_name) as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for line in csv_reader:
        #         line_split = line[0].split()
        #         preds = [label2index[l] for l in line_split]
        #         row = [data[iter][0], data[iter][1], np.asarray(preds)]

        final_out.append(row)
    file = open('align_results/split_{}/stage_1/nn_viterbi_stage_1_vals_train'.format(str(args.split_id)), 'wb')
    pickle.dump(final_out, file)
    file.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

