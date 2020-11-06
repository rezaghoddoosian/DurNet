# Reference:
# https://github.com/Zephyr-D/TCFPN-ISBA
from __future__ import division

from keras.models import Model
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras import backend as K
from keras.optimizers import rmsprop
import cv2
import numpy as np
import time

def mask_data(X, Y, max_len=None, mask_value=0):
    if len(X)!=0:
        X_ = np.zeros([len(X), max_len, X[0].shape[1]]) + mask_value
    else:
        X_=None
    if len(Y)!=0:
        Y_ = np.zeros([len(Y), max_len, Y[0].shape[1]]) + mask_value
    else:
        Y_=None
    mask = np.zeros([max(len(X),len(Y)), max_len])
    for i in range(max(len(X),len(Y))):
        if len(X)!=0:
            l = X[i].shape[0]
            X_[i, :l] = X[i]

        else:
            l = Y[i].shape[0]

        if len(Y) != 0:
            Y_[i, :l] = Y[i]

        mask[i, :l] = 1
    return X_, Y_, mask[:, :, None]


# Unmask data
def unmask(X, M):
    if X[0].ndim == 1 or (X[0].shape[0] > X[0].shape[1]):
        return [X[i][M[i].flatten() > 0] for i in range(len(X))]
    else:
        return [X[i][:, M[i].flatten() > 0] for i in range(len(X))]


def TCFPN(n_nodes, conv_len, n_classes, n_feat, in_len,
          optimizer=rmsprop(lr=1e-4), return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(in_len, n_feat))
    model = inputs
    lyup = []
    lydown = []

    # ---- Encoder ----
    for i in range(n_layers):
        model = Conv1D(n_nodes[i], conv_len, padding='same', use_bias=False)(model)
        model = BatchNormalization()(model)
        model = SpatialDropout1D(0.1)(model)
        model = Activation('relu')(model)
        model = MaxPooling1D(2, padding='same')(model)
        lyup.append(model)

    # ---- Decoder ----
    model = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    modelout = UpSampling1D(8)(modelout)
    lydown.append(modelout)

    model = UpSampling1D(2)(model)
    res = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(lyup[-2])
    model = add([model, res])
    model = Conv1D(n_nodes[0], conv_len, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    modelout = UpSampling1D(4)(modelout)
    lydown.append(modelout)

    model = UpSampling1D(2)(model)
    res = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(lyup[-3])
    model = add([model, res])
    model = Conv1D(n_nodes[0], conv_len, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    modelout = UpSampling1D(2)(modelout)
    lydown.append(modelout)

    model = UpSampling1D(2)(model)
    res = Conv1D(n_nodes[0], 1, padding='same', use_bias=False)(inputs)
    model = add([model, res])
    model = Conv1D(n_nodes[0], conv_len, padding='same', use_bias=False)(model)
    modelout = SpatialDropout1D(0.1)(model)
    modelout = TimeDistributed(Dense(n_classes, name='fc', activation='softmax'))(modelout)
    lydown.append(modelout)

    model = average(lydown)

    model = Model(inputs=inputs, outputs=model)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  sample_weight_mode="temporal")

    if return_param_str:
        param_str = "TCFPN_C{}_L{}".format(conv_len, n_layers)
        return model
    else:
        return model




def test_allignment(AP_testraw,y_test_real,y_test_seq):
    split_start = time.time()
    z = 1  # how many label instance to insert
    u = 0.02  # threshold
    ran = 0.1  # randomness
    for test_align in range(10):  # Here we do alignmnet for the test set, but model is not trained anymore. Run for 10 rounds and insertion happens in each round

        y_test_seqn = []

        for i in range(len(AP_testraw)):
            # x = X_test_m[i]
            pmp = AP_testraw[i]  #VisModelTempPred_test

            # realignment
            seq = y_test_seq[i]
            seqn = list(seq[:])
            k = 0
            inds = np.arange(1, len(seq)) / len(seq) * len(y_test_real[i]) #per_sec_y_test
            inds = inds.astype(np.int)

            for ind in range(len(inds)):
                if seq[ind] != seq[ind + 1]:
                    if pmp[inds[ind], seq[ind]] > pmp[inds[ind], seq[ind + 1]] + u:
                        for zz in range(z):
                            rr = np.random.random()
                            if rr > ran:
                                seqn.insert(ind + k + 1, seq[ind])
                            else:
                                seqn.insert(ind + k + 1, seq[ind + 1])
                        k += z
                    elif pmp[inds[ind], seq[ind]] < pmp[inds[ind], seq[ind + 1]] - u:
                        for zz in range(z):
                            rr = np.random.random()
                            if rr > ran:
                                seqn.insert(ind + k + 1, seq[ind + 1])
                            else:
                                seqn.insert(ind + k + 1, seq[ind])
                        k += z
            y_test_seqn.append(seqn)

        y_test_seq = y_test_seqn[:]
    y_test_temp = [cv2.resize(np.array(i), (1, len(j)), interpolation=cv2.INTER_NEAREST).reshape(len(j))for i, j in zip(y_test_seq, y_test_real)]
    split_end = time.time()
    print('Time elapsed for TCFPN in ms:', (split_end - split_start) * 1000)
    return y_test_temp
