
import numpy as np
from utils.dataset import Dataset
from utils.network import Trainer, Forwarder
from utils.viterbi import Viterbi
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', default=4, type=int)
import sys
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
    with open('data/split{}.train'.format(str(args.split_id)), 'r') as f:
        video_list = f.read().split('\n')[0:-1]
    dataset = Dataset('data', video_list, label2index, shuffle = True)
    print('done')

    ### generate path grammar for inference ########################################
    paths = set()
    for _, transcript, _ in dataset:
        paths.add(' '.join([index2label[index] for index in transcript]))
    with open('align_results/grammar_split_{}.txt'.format(args.split_id), 'w') as f:
        f.write('\n'.join(paths) + '\n')
    print('done')

    ### actual nn-viterbi training #################################################
    decoder = Viterbi(None, None, frame_sampling = 30, max_hypotheses = np.inf) # (None, None): transcript-grammar and length-model are set for each training sequence separately, see trainer.train(...)
    trainer = Trainer(decoder, dataset.input_dimension, dataset.n_classes, buffer_size = len(dataset), buffered_frame_ratio = 25)
    learning_rate = 0.01

    load_iteration = 10000
    forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
    forwarder.load_model('align_results/split_{}/stage_1/network.iter-'.format(str(args.split_id)) + str(load_iteration) + '.net')
    print('Loading Model Done')

    fname = 'align_results/new_pseudo_gt_split{}_train.npy'.format(str(args.split_id))
    new_data = np.load(fname,allow_pickle=True)
    new_pseudo_gt = []
    # train for 10000 iterations
    i = 0
    all_video_list = []
    end_flag = 0
    max_epoch = 2
    # for i in range(1456):
    while end_flag == 0:
        i += 1
        sequence, transcript, videoname = dataset.get()
        for iter in range(len(new_data)):
            if videoname == new_data[iter][0]:
                new_pseudo_gt = new_data[iter][1]
                break
        all_video_list.append(videoname)
        vid_count = 0
        for v_iter in range(len(all_video_list)):
            if all_video_list[v_iter] == videoname:
                vid_count += 1
            if vid_count == max_epoch:
                print('duplicate video found')
                end_flag = 1
        print(i, videoname)
        loss = trainer.train(sequence, transcript, videoname, new_pseudo_gt, batch_size = 1024, learning_rate = learning_rate)
        # print some progress information
        if (i+1) % 100 == 0:
            print('Iteration %d, loss: %f' % (i+1, loss))
    #save model every 1000 iterations

    i += 10000
    print('save snapshot ' + str(i+1))
    network_file = 'align_results/split_{}/stage_2/network.iter-'.format(str(args.split_id)) + str(i + 1) + '.net'
    length_file = 'align_results/split_{}/stage_2/lengths.iter-'.format(str(args.split_id)) + str(i + 1) + '.txt'
    prior_file = 'align_results/split_{}/stage_2/prior.iter-'.format(str(args.split_id)) + str(i + 1) + '.txt'
    trainer.save_model(network_file, length_file, prior_file)

    # sys.stdout = orig_stdout
    # f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
