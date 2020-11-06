import pickle
import numpy as np
import csv
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_id', default=2, type=int)
import sys
#The code is obtained from https://github.com/alexanderrichard/NeuralNetwork-Viterbi and is modified accordingly.
def main(args):

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

    def frame_accuracy(test,gt,prev_pseudo_gt):
        t = [np.sum(test[i] == gt[i])  for i in range(len(test))]
        T = [len(test[i]) for i in range(len(test))]
        frame_accuracy=np.sum(t)/np.sum(T)
        t2 = [np.sum(prev_pseudo_gt[i] == gt[i])  for i in range(len(test))]
        frame_accuracy2=np.sum(t2)/np.sum(T)
        per_vid_frame_accuracy1 = np.asarray([np.sum(test[i] == gt[i]) / len(test[i]) for i in range(len(test))])
        per_vid_frame_accuracy2 = np.asarray([np.sum(prev_pseudo_gt[i] == gt[i]) / len(prev_pseudo_gt[i]) for i in range(len(prev_pseudo_gt))])
        # print(frame_accuracy2)
        print('Previous framelevel acc with background:{}'.format(np.average(per_vid_frame_accuracy2)))
        # print(frame_accuracy2)
        print("#####")
        # print(np.average(per_vid_frame_accuracy1))
        print('Current framelevel acc with background:{}'.format(np.average(per_vid_frame_accuracy1)))
        # print(frame_accuracy)
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

    def map_index(prev,map_list):
        new=np.zeros([len(prev)],dtype=int)
        for i in range(1,48):
            ind= prev==i
            if sum(ind)!=0:
                new[ind]=int(map_list[i])

        return new

    def convert_and_calculate_metrics(pred_data,gt,base_line_in):
        # pred_data: a list of lists :0-filename, 1-numpy array.              each member is for a video (new pseudo_gt)  [labels should be converted]
        # base_line: np array, each element is another array annotating frame level labels for each video   [labels are OK]
        # gt: np array, each element is another array annotating frame level labels for each video  [labels are OK]

        predicted = []
        base_line = []
        maplist = []
        actions = open("act2actmap.txt").readlines()
        for i in range(48):
            maplist.append(actions[i].split()[1])   #mapping baseline to mine

        for v in range(len(pred_data)): #conversion
            # predicted.append(np.asarray(pred_data[v][1])[4::])
            predicted.append(np.asarray(pred_data[v][1]))
            base_line.append(np.asarray(base_line_in[v]))
            # predicted[-1] = map_index(predicted[-1], maplist)
            # base_line[-1] = map_index(base_line[-1], maplist)

        predicted = np.asarray(predicted)
        frame_accuracy(predicted, gt, base_line)

        frame_accuracy_w_bg(predicted, gt, base_line)
        iou = IoU(list(predicted), gt)
        print("Current-IOU: " + str(iou))
        iod =IoD(list(predicted), gt)
        print("Current-IOD: " + str(iod))
        iou = IoU(list(base_line), gt)
        print("Previous-IOU: " + str(iou))
        iod = IoD(list(base_line), gt)
        print("Previous-IOD: " + str(iod))


    fname = 'align_results/gt_test_withfnames_correct_{}'.format(str(args.split_id))
    with open(fname, 'rb') as in_f:
        gt = pickle.load(in_f)

    label2index = dict()
    index2label = dict()
    with open('data/mapping.txt', 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    stage1_results = []
    stage1_results_only_preds = []
    result_folder = 'align_results/split_{}/stage_1/viterbi_outputs_align_test_split/'.format(str(args.split_id))
    for iter in range(len(gt)):
        sample_name = gt[iter][0]
        filename = result_folder + sample_name
        row = []

        with open(result_folder + sample_name, 'rb') as in_f:
            preds = pickle.load(in_f)
            # preds = preds[4::]
            row = [sample_name, np.asarray(preds)]

        # with open(result_folder + sample_name) as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for line in csv_reader:
        #         line_split = line[0].split()
        #         preds = [label2index[l] for l in line_split]
        #         preds = preds[4::]
        #         row = [sample_name, np.asarray(preds)]
        stage1_results.append(row)
        stage1_results_only_preds.append(row[1])

    stage2_results = []
    stage2_results_only_preds = []
    gt_only_preds = []
    result_folder = 'align_results/split_{}/stage_2/viterbi_outputs_align_test_split/'.format(str(args.split_id))
    for iter in range(len(gt)):
        sample_name = gt[iter][0]
        row = []
        with open(result_folder + sample_name, 'rb') as in_f:
            preds = pickle.load(in_f)
            # preds = preds[4::]
            row = [sample_name, np.asarray(preds)]

        # with open(result_folder + sample_name) as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     for line in csv_reader:
        #         line_split = line[0].split()
        #         preds = [label2index[l] for l in line_split]
        #         preds = preds[4::]
        #         row = [sample_name, np.asarray(preds)]

        # print(len(row[1]), len(gt[iter][1]))
        stage2_results.append(row)
        stage2_results_only_preds.append(row[1])
        gt_only_preds.append(gt[iter][1])

    stage2_results_only_preds = np.asarray(stage2_results_only_preds)
    gt_only_preds = np.asarray(gt_only_preds)

    print('############# Alignmnent Stage 1 and 2 Results for split {} ############'.format(args.split_id))

    convert_and_calculate_metrics(stage2_results, gt_only_preds, stage1_results_only_preds)

    # sys.stdout = orig_stdout
    # f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)