import tensorflow as tf
import numpy as np
import metrics

class LengthModule:
    def __init__(self,graph,nClasses,nActions,nObjects,length,video_info_size,video_obj_info_size,pre_f_size,h_size,emb_size,num_layers,feature_size,batch_size,step,duration=30,sample_rate=2,hard_label=False):
        self.graph=graph
        self.duration=duration
        self.sample_rate=sample_rate
        self.pre_f_size = pre_f_size
        self.training = tf.placeholder(tf.bool, name='phase_train')
        self.keep_p = tf.placeholder(tf.float32)
        self.feature_size = feature_size
        self.nClasses = nClasses
        self.nActions=nActions
        self.nObjects=nObjects
        self.h_size = h_size
        self.video_info_size=video_info_size
        self.video_obj_info_size = video_obj_info_size
        self.emb_size=emb_size
        self.length=length
        self.step=step
        self.max_length = duration//step
        self.num_layers = num_layers
        self.Xf_placeholder = tf.placeholder(tf.float32, shape=(None, self.max_length, self.feature_size), name='bacth_in_features')
        self.Xa_placeholder = tf.placeholder(tf.int64, shape=(None),name='bacth_in_action')
        self.Xo_placeholder_hot = tf.placeholder(tf.float32, shape=(None,nObjects), name='bacth_in_objects')
        self.V_I_placeholder_hot = tf.placeholder(tf.float32, shape=(None, video_info_size), name='bacth_in_VI')
        self.V_O_placeholder_hot = tf.placeholder(tf.float32, shape=(None, video_obj_info_size), name='bacth_in_VI')
        self.X48a_placeholder = tf.placeholder(tf.int64, shape=(None), name='bacth_in_48action')
        self.NSeg_placeholder = tf.placeholder(tf.int64, shape=(None), name='bacth_in_nSeg')
        self.batch_size = batch_size
        self.x = np.asarray([5 * i + 3 for i in range(0, length)])
        self.hard_label=hard_label


    def create_categorical_map_label(self, action2object, y_train,n_objects):
        inp = np.zeros([len(y_train),1, n_objects])
        for i in range(len(y_train)): #video i
            for a in y_train[i]:
                if  action2object[a]!=n_objects:
                        basic_a = action2object[a]
                        inp[i,0,basic_a] = 1
        return inp

    def create_categorical_label(self, y_train,n_classes):
        inp = np.zeros([len(y_train), n_classes])
        for i in range(len(y_train)): #video i
            if y_train[i]==list:
                for a in y_train[i]:
                    if y_train[i] < n_classes:
                        inp[i,a] = 1
            else:
                if y_train[i]<n_classes:
                    inp[i,y_train[i]]=1
        return inp

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

    def hinge_loss(self,one_hot_labels, logits,weight_per_label):
        delta=1.0
        true_logits=tf.reduce_sum(one_hot_labels*logits,axis=-1)
        true_logits=tf.expand_dims(true_logits,axis=-1)
        diff=logits-true_logits
        loss=0
        D=tf.shape(diff)[-1]
        loss=tf.math.maximum(0.0,diff + delta)
        hv_bool=tf.cast(one_hot_labels, tf.bool)
        nhv_bool=tf.math.logical_not(hv_bool)
        loss=loss*tf.cast(nhv_bool,tf.float32)
        loss=tf.reduce_sum(loss,axis=-1)
        loss=tf.expand_dims(loss,axis=-1)
        loss=loss*weight_per_label
        loss=tf.reduce_mean(loss)
        return loss



    def Soft_Label(self,x,y,length):
        def Gaussian(x, mu, sig): #for soft labeling
            return np.exp(-((x - mu) ** 2) / (2 * (sig ** 2)))

        mu=y
        soft_lb=np.ones([length])*Gaussian(x,mu,2)
        soft_lb=soft_lb/np.sum(soft_lb)

        assert np.sum(soft_lb)>0.999999999 and np.sum(soft_lb)<1.000000001 ,"not a pdf "
        return soft_lb

    def create_label(self,label1,action2atomic_action,gt):
        label = np.copy(label1)
        for a in gt:
            basic_a=action2atomic_action[a]
            label[basic_a-1]=1
        return label


    def create_adaptive_soft_labels(self,action2step_map, Y,n_length,X_actions):  #Based on the type of the action creates corresponding soft labels with different granalarities.
        Y_prime=[]
        for y,a in zip(Y,X_actions):
            step=action2step_map[a]
            max_length=step*n_length
            if y>max_length:
                y=max_length
            act_idx=(y-1)//step
            assert act_idx<n_length and act_idx>-1
            sof_lb=self.Soft_Label(self.x, act_idx * 5 + 3, n_length)
            Y_prime.append(sof_lb)

        Y_prime=np.asarray(Y_prime)
        return Y_prime


    def rescale_N(self,action2step_map, N,n_length,X_actions):  #Based on the type of the action rescale the past time with different granalarities.
        N_prime=[]
        for n,a in zip(N,X_actions):
            time_behind=n*self.sample_rate
            step=action2step_map[a]
            max_length=step*n_length
            if time_behind>max_length:
                time_behind=max_length
            if time_behind==0:
                time_behind=1
            act_idx=(time_behind-1)//step
            N_prime.append(act_idx)

        N_prime=np.asarray(N_prime)
        return N_prime

    def generate_X_Y(self,data,predicted_actions,action2atomic_action,action2object_map,shuffle,gt,per_vid_acts,per_vid_obj_acts):
        #data: a list of [#frame,feature_size] arrays per video
        # predicted_actions: a list of 1 D (#seconds in vid) arrays per video
        # training if true , it shuffles the data
        def unison_shuffled_copies(a, b,c,d,e,g,h,i):
            p = np.random.permutation(len(a))
            return a[p], b[p], c[p], d[p],e[p],g[p],h[p],i[p]
        V_O_info=[]
        V_act_info=[]
        X_features = []
        X_actions = []
        X_base_actions=[]
        X_objects = []
        Y = []
        N=[]
        Sec_Ahead=[]
        for vid_num,vid in enumerate(predicted_actions):


            action_labels=vid[::self.sample_rate]
            action_labels_true = gt[vid_num][::self.sample_rate]
            length=len(action_labels)
            indices=[]
            temp=vid[1:]-vid[0:-1]
            idx=np.where(temp!=0)
            if np.sum(idx)==0:
                idx=np.asarray([len(vid)])
            else:
                idx=idx[0]+1
                idx=np.append(idx,len(vid))
            i=0

            for n in range(length):



                if n==0:
                    n_prev_seg = 0
                else:
                    if action_labels[n]!=action_labels[n-1]:
                        n_prev_seg=0
                    else:
                        n_prev_seg=n_prev_seg+1


                V_O_info.append(per_vid_obj_acts[vid_num,:])
                V_act_info.append(per_vid_acts[vid_num,:])
                X_objects.append(action2object_map[action_labels_true[n]])
                X_actions.append(action_labels[n])
                indices.append(self.sample_rate * n)  # the index in the original time scale (1/15 second scale)
                N.append(n_prev_seg)
                X_base_actions.append(action2atomic_action[action_labels_true[n]])
                # idx =index in the 1 second scale
                if (idx[i]-self.sample_rate*n)>0:
                    y=idx[i] - self.sample_rate * n
                else:
                    while (idx[i]-self.sample_rate*n)<=0:
                        i=i+1
                        y=idx[i] - self.sample_rate * n

                if y==0:
                    y=1
                    assert 1==0,"y==0 found"
                if self.hard_label:

                    Y.append(y)
                else:

                    if n_prev_seg==0:
                        Sec_Ahead.append([action2atomic_action[action_labels[n]],y])

                    Y.append(y)

            for n,index in enumerate(indices):
                if index+self.duration>len(data[vid_num]):
                    x = data[vid_num][index::self.step, :]
                    x=np.concatenate((x,np.zeros([self.max_length-len(x),self.feature_size])),axis=0)

                else:
                    x=data[vid_num][index:index+self.duration:self.step,:]

                X_features.append(x)
        V_O_info=np.asarray(V_O_info)
        V_act_info=np.asarray(V_act_info)
        X_actions= np.array(X_actions)
        X_base_actions = np.array(X_base_actions)
        X_objects = np.array(X_objects)
        X_features= np.array(X_features)
        Y = np.array(Y)
        N=np.asarray(N)
        Sec_Ahead=np.asarray(Sec_Ahead)


        if shuffle:
            X_base_actions,X_actions,X_features,Y,N,X_objects,V_act_info,V_O_info= unison_shuffled_copies(X_base_actions,X_actions , X_features,Y,N,X_objects,V_act_info,V_O_info)
        del data
        del predicted_actions
        dataset_for_length_prediction=[X_features,X_base_actions,Y,N,Sec_Ahead,X_actions,X_objects,V_act_info,V_O_info]
        return dataset_for_length_prediction



    def model(self):
        self.X48a_placeholder_hot = tf.one_hot(self.X48a_placeholder, self.nActions)
        self.Xa_placeholder_hot = tf.one_hot(self.Xa_placeholder, self.nClasses)
        self.NSeg_placeholder_hot = tf.one_hot(self.NSeg_placeholder, self.length)
        batch_size = tf.shape(self.Xa_placeholder)[0]
        def batchNorm(x, beta, gamma, training, scope='bn'):
            with tf.variable_scope(scope):
                batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
                ema = tf.train.ExponentialMovingAverage(decay=0.5)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(batch_mean), tf.identity(batch_var)

                mean, var = tf.cond(training, mean_var_with_update,
                                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
                normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
            return normed
        with tf.variable_scope('pre_fc',reuse=tf.AUTO_REUSE):
            pre_fc_weights = tf.get_variable('weights', [self.feature_size,self.pre_f_size], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,dtype=tf.float32))
            pre_fc_biases = tf.get_variable('biases', [self.pre_f_size], dtype=tf.float32,
                                             initializer=tf.constant_initializer(0.0))

            reshaped_input = tf.reshape(self.Xf_placeholder, [-1, self.feature_size])
            input_RNN = tf.matmul(reshaped_input, pre_fc_weights)

            input_RNN = batchNorm(input_RNN, pre_fc_biases, None, self.training, scope='bn')
            input_RNN = tf.nn.relu(input_RNN)
            input_RNN = tf.reshape(input_RNN, [-1,self.max_length, self.pre_f_size])
            input_RNN = tf.nn.dropout(input_RNN, self.keep_p)


        with tf.variable_scope("LSTM", reuse=tf.AUTO_REUSE):
            enc_fw_cells =[tf.nn.rnn_cell.LSTMCell(self.h_size) for i in range(self.num_layers)]
            enc_bw_cells = [tf.nn.rnn_cell.LSTMCell(self.h_size) for i in range(self.num_layers)]
            (outputs, fw_state, bw_state) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=enc_fw_cells, cells_bw=enc_bw_cells, inputs=input_RNN, dtype=tf.float32)
            outputs=tf.unstack(tf.transpose(outputs, [1, 0, 2]))
            seconds=self.duration//15
            two_seconds=seconds//2
            self.out_idx=np.linspace(0,self.max_length,two_seconds+2,dtype=int)[1:]



        OUT=0
        for i,idx in enumerate(self.out_idx):
            with tf.variable_scope('out_fc_%d' %i,reuse=tf.AUTO_REUSE):
                out_fc_weights = tf.get_variable('weights', [self.h_size*2, self.emb_size], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))
                out_fc_biases = tf.get_variable('out_biases', [self.emb_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            OUT=tf.matmul(outputs[int(idx)-1], out_fc_weights)+OUT


        OUT = batchNorm(OUT, out_fc_biases, None, self.training, scope='bn')
        OUT=tf.nn.relu(OUT)
        OUT= tf.nn.dropout(OUT, self.keep_p)

        with tf.variable_scope('post_fc',reuse=tf.AUTO_REUSE):
            post_fc_weights = tf.get_variable('weights', [self.emb_size, self.emb_size], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,dtype=tf.float32))

        with tf.variable_scope('action_fc1',reuse=tf.AUTO_REUSE):
            action_fc_weights1 = tf.get_variable('weights', [self.nClasses,self.emb_size], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,dtype=tf.float32))

            action_fc_biases1 = tf.get_variable('class_biases', [self.emb_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            action_fc1_out = tf.matmul(self.Xa_placeholder_hot, action_fc_weights1)
            action_fc1_out = batchNorm(action_fc1_out, action_fc_biases1, None, self.training, scope='bn')
            action_fc1_out = tf.nn.relu(action_fc1_out)
            action_fc1_out = tf.nn.dropout(action_fc1_out, self.keep_p)

        with tf.variable_scope('action_fc2',reuse=tf.AUTO_REUSE):
            action_fc_weights2 = tf.get_variable('weights', [self.emb_size,self.emb_size], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,dtype=tf.float32))

            emb_biases = tf.get_variable('class_biases', [self.emb_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))

        with tf.variable_scope('segment_fc1', reuse=tf.AUTO_REUSE):
            segment_fc_weights1 = tf.get_variable('weights', [self.length, self.emb_size], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))

            segment_fc_biases1 = tf.get_variable('class_biases', [self.emb_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            segment_fc1_out = tf.matmul(self.NSeg_placeholder_hot, segment_fc_weights1)
            segment_fc1_out = batchNorm(segment_fc1_out, segment_fc_biases1, None, self.training, scope='bn')
            segment_fc1_out = tf.nn.relu(segment_fc1_out)
            segment_fc1_out = tf.nn.dropout(segment_fc1_out, self.keep_p)

        with tf.variable_scope('segment_fc2',reuse=tf.AUTO_REUSE):
            segment_fc_weights2 = tf.get_variable('weights', [self.emb_size,self.emb_size], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,dtype=tf.float32))

        length_emb=tf.matmul(OUT,post_fc_weights)+tf.matmul(action_fc1_out, action_fc_weights2)+tf.matmul(segment_fc1_out, segment_fc_weights2)
        length_emb = batchNorm(length_emb,emb_biases, None, self.training, scope='bn')
        length_emb = tf.nn.relu(length_emb)
        length_emb = tf.nn.dropout(length_emb, self.keep_p)
        with tf.variable_scope('length_classification', reuse=tf.AUTO_REUSE):
            classification_weights = tf.get_variable('weights', [self.emb_size, self.length ], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))


            classification_biases = tf.get_variable('biases', [self.length], dtype=tf.float32,initializer=tf.constant_initializer(0.0))

        self.predicted_length = tf.matmul(length_emb,classification_weights)+classification_biases
        self.out_probability = tf.nn.softmax(self.predicted_length)
        #######################base_action classification section#########################
        with tf.variable_scope('V_I_emb', reuse=tf.AUTO_REUSE):
            V_I_weights = tf.get_variable('weights', [self.video_info_size, self.emb_size], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))
            V_I_biases = tf.get_variable('class_biases', [self.emb_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))

        base_action_fc1_out = tf.matmul(self.V_I_placeholder_hot, V_I_weights)
        base_action_fc1_out = batchNorm(base_action_fc1_out, V_I_biases, None, self.training, scope='bn')
        base_action_fc1_out = tf.nn.relu(base_action_fc1_out)
        base_action_fc1_out= tf.nn.dropout(base_action_fc1_out, self.keep_p)

        with tf.variable_scope('base_post_fc', reuse=tf.AUTO_REUSE):
            base_post_fc_weights = tf.get_variable('weights', [self.emb_size, self.emb_size], dtype=tf.float32,
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))
            base_emb_biases = tf.get_variable('class_biases', [self.emb_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))

        with tf.variable_scope('base_action_fc', reuse=tf.AUTO_REUSE):
            base_action_fc_weights = tf.get_variable('weights', [self.emb_size, self.emb_size], dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))

        base_emb = tf.matmul(OUT, base_post_fc_weights) + tf.matmul(base_action_fc1_out, base_action_fc_weights)
        base_emb = batchNorm(base_emb, base_emb_biases, None, self.training, scope='bn')
        base_emb = tf.nn.relu(base_emb)
        base_emb = tf.nn.dropout(base_emb, self.keep_p)
        with tf.variable_scope('base_classification', reuse=tf.AUTO_REUSE):
            base_classification_weights = tf.get_variable('weights', [self.emb_size, self.nClasses], dtype=tf.float32,
                                                          initializer=tf.contrib.layers.xavier_initializer(
                                                              uniform=False, seed=None, dtype=tf.float32))

            base_classification_biases = tf.get_variable('class_biases', [self.nClasses], dtype=tf.float32,
                                                         initializer=tf.constant_initializer(0.0))

        self.base_out = tf.matmul(base_emb, base_classification_weights) + base_classification_biases
        self.base_out_probability = tf.nn.softmax(self.base_out)
        #######################Object Classification Section#########################
        with tf.variable_scope('object_pre_fc', reuse=tf.AUTO_REUSE):
            object_pre_fc_weights = tf.get_variable('weights', [self.feature_size, self.pre_f_size], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                     seed=None,
                                                                                                     dtype=tf.float32))
            object_pre_fc_biases = tf.get_variable('biases', [self.pre_f_size], dtype=tf.float32,
                                                   initializer=tf.constant_initializer(0.0))

            reshaped_input = tf.reshape(self.Xf_placeholder, [-1, self.feature_size])
            input_RNN = tf.matmul(reshaped_input, object_pre_fc_weights)
            input_RNN = batchNorm(input_RNN, object_pre_fc_biases, None, self.training, scope='bn')
            input_RNN = tf.nn.relu(input_RNN)
            input_RNN = tf.reshape(input_RNN, [-1, self.max_length, self.pre_f_size])




        with tf.variable_scope("object_LSTM", reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.LSTMCell(self.h_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_p)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 1, state_is_tuple=True)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, input_RNN, initial_state=init_state, time_major=False)
            outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

        with tf.variable_scope('vid2obj', reuse=tf.AUTO_REUSE):
            V_O_weights = tf.get_variable('weights', [self.video_obj_info_size , self.emb_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))
            V_O_biases = tf.get_variable('out_biases', [self.emb_size], dtype=tf.float32,initializer=tf.constant_initializer(0.0))

        V_O_fc_out = tf.matmul(self.V_O_placeholder_hot,V_O_weights)
        V_O_fc_out = batchNorm(V_O_fc_out, V_O_biases, None, self.training, scope='bn')
        V_O_fc_out= tf.nn.relu(V_O_fc_out)
        V_O_fc_out = tf.nn.dropout(V_O_fc_out, self.keep_p)

        with tf.variable_scope('vid2obj_fc', reuse=tf.AUTO_REUSE):
            V_O_weights = tf.get_variable('weights', [ self.emb_size , self.emb_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))

        V_O_fc_out = tf.matmul( V_O_fc_out, V_O_weights)

        with tf.variable_scope('object_out_fc', reuse=tf.AUTO_REUSE):
            object_out_fc_weights = tf.get_variable('weights', [self.h_size, self.emb_size], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                     seed=None,
                                                                                                     dtype=tf.float32))
            object_out_fc_biases = tf.get_variable('out_biases', [self.emb_size], dtype=tf.float32,
                                                   initializer=tf.constant_initializer(0.0))

        obj_OUT = tf.matmul(outputs[-1], object_out_fc_weights)+V_O_fc_out
        obj_OUT = batchNorm(obj_OUT, object_out_fc_biases, None, self.training, scope='bn')
        obj_OUT = tf.nn.relu(obj_OUT)
        obj_OUT = tf.nn.dropout(obj_OUT, self.keep_p)

        with tf.variable_scope('object_post_fc', reuse=tf.AUTO_REUSE):
            object_post_fc_weights = tf.get_variable('weights', [self.emb_size, self.emb_size], dtype=tf.float32,
                                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                      seed=None,
                                                                                                      dtype=tf.float32))
            object_emb_biases = tf.get_variable('class_biases', [self.emb_size], dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.0))

        obj_emb = tf.matmul(obj_OUT, object_post_fc_weights)
        obj_emb = batchNorm(obj_emb, object_emb_biases, None, self.training, scope='bn')
        obj_emb = tf.nn.relu(obj_emb)
        obj_emb = tf.nn.dropout(obj_emb, self.keep_p)
        with tf.variable_scope('object_classification', reuse=tf.AUTO_REUSE):
            object_classification_weights = tf.get_variable('weights', [self.emb_size, self.nObjects], dtype=tf.float32,
                                                            initializer=tf.contrib.layers.xavier_initializer(
                                                                uniform=False, seed=None, dtype=tf.float32))
            object_classification_biases = tf.get_variable('class_biases', [self.nObjects], dtype=tf.float32,
                                                           initializer=tf.constant_initializer(0.0))

        with tf.variable_scope('base2object', reuse=tf.AUTO_REUSE):
            base2object_weights = tf.get_variable('weights', [self.emb_size, self.emb_size], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                   seed=None,
                                                                                                   dtype=tf.float32))
            base2object_biases = tf.get_variable('class_biases', [self.emb_size], dtype=tf.float32,
                                                 initializer=tf.constant_initializer(0.0))
        base2object = tf.matmul(base_emb, base2object_weights)
        base2object = batchNorm(base2object, base2object_biases, None, self.training, scope='bn')
        base2object = tf.math.sigmoid(base2object)

        obj_emb = (obj_emb * base2object)
        self.object_out = tf.matmul(obj_emb, object_classification_weights) + object_classification_biases
        self.object_out_probability = tf.nn.softmax(self.object_out)



    def train(self,First_Time,training_pack, val_pack,class_weights, keep_p,var_save_path, num_epochs, lr, regul,object_lr):

            [data_f_tr,data_a_tr,labels_tr,nSeg_tr,_48action_lb,object_lb,V_A,V_O]=training_pack
            [data_f_val,data_a_val,labels_val,nSeg_val,_48action_lb_val,object_lb_val,V_A_val,V_O_val]=val_pack
            [class_weight,base_class_weight,object_class_weight]=class_weights

            def get_weighted_loss(weights,y_true, y_pred):
                e=0.00000000000000000000001
                y_pred = tf.clip_by_value(y_pred, e, 1 - e)
                mt_loss= - y_true * tf.log(y_pred) - (1 - y_true) * tf.log(1 - y_pred)
                weighted_loss=tf.reduce_mean(tf.reduce_sum((weights[:, :, 0] ** (1 - y_true)) * (weights[:, :, 1] ** (y_true)) * mt_loss,axis=-1))
                return weighted_loss

            def save_variables(sess, path):
                saver = tf.train.Saver()
                print('saving variables...\n')
                saver.save(sess, path + 'my_model')

            def batch_gen(data_f,data_a, label,segment,_48acts,objs,v_i,v_o,batch_size):  # data=[Total data points, T,F]   #label=[Total Data Points, 1]
                n = len(data_a)
                batch_num = n // batch_size
                for b in range(batch_num):  # Here it generates batches of data within 1 epoch consecutively
                    X_f = data_f[batch_size * b:batch_size * (b + 1), :, :]
                    X_a = data_a[batch_size * b:batch_size * (b + 1)]
                    N=segment[batch_size * b:batch_size * (b + 1)]
                    X_48a = _48acts[batch_size * b:batch_size * (b + 1)]
                    X_o = objs[batch_size * b:batch_size * (b + 1), :]
                    V_I = v_i[batch_size * b:batch_size * (b + 1), :]
                    V_O = v_o[batch_size * b:batch_size * (b + 1), :]
                    if self.hard_label:
                        Y = label[batch_size * b:batch_size * (b + 1)]
                    else:
                        Y = label[batch_size * b:batch_size * (b + 1),:]
                    yield X_f,X_a, Y,N,X_48a,X_o,V_I,V_O
                if n > batch_size * (b + 1):
                    X_f = data_f[batch_size * (b + 1):, :, :]
                    X_a = data_a[batch_size * (b + 1):]
                    N = segment[batch_size * (b + 1):]
                    X_48a = _48acts[batch_size * (b + 1):]
                    X_o = objs[batch_size * (b + 1):, :]
                    V_I = v_i[batch_size * (b + 1):, :]
                    V_O = v_o[batch_size * (b + 1):, :]
                    if self.hard_label:
                        Y = label[batch_size * (b + 1):]
                    else:
                        Y = label[batch_size * (b + 1):,:]

                    yield X_f,X_a, Y,N,X_48a,X_o,V_I,V_O

            def epoch_gen(training_pack,batch_size,num_epochs):  # data=[Total data points, T,F]  # This generates epochs of batches of data
                [data_f, data_a,label, segment,_48acts,objs,v_i,v_o] = training_pack
                for epoch in range(num_epochs):  # Inside one epoch
                    yield batch_gen(data_f, data_a, label, segment,_48acts,objs,v_i,v_o, batch_size)

            # if First_Time is True, the weights are initialized
            if First_Time:
                C=0
                with tf.variable_scope('pre_fc', reuse=True):
                    pre_fc_weights = tf.get_variable('weights')
                with tf.variable_scope('action_fc1', reuse=True):
                    action_fc_weights1 = tf.get_variable('weights')
                with tf.variable_scope('action_fc2', reuse=True):
                    action_fc_weights = tf.get_variable('weights')

                with tf.variable_scope('post_fc', reuse=True):
                    post_fc_weights = tf.get_variable('weights')

                with tf.variable_scope('segment_fc1', reuse=True):
                    segment_fc_weights1 = tf.get_variable('weights')
                with tf.variable_scope('segment_fc2', reuse=True):
                    segment_fc_weights = tf.get_variable('weights')
                with tf.variable_scope('length_classification', reuse=True):
                    length_class_weights = tf.get_variable('weights')

                for i in range(len(self.out_idx)):
                    with tf.variable_scope('out_fc_%d' % i,reuse=True):
                        C= tf.nn.l2_loss(tf.get_variable('weights'))+C

                with tf.variable_scope('base_post_fc', reuse=True):
                    base_post_fc_weights = tf.get_variable('weights')
                with tf.variable_scope('base_action_fc', reuse=True):
                    base_action_fc2_weights = tf.get_variable('weights')
                with tf.variable_scope('base_classification', reuse=True):
                    base_class_weights = tf.get_variable('weights')
                object_reg=0
                with tf.variable_scope('object_pre_fc', reuse=True):
                    object_pre_fc_weights = tf.get_variable('weights')
                    object_reg= object_reg+tf.nn.l2_loss(object_pre_fc_weights)
                with tf.variable_scope('object_out_fc', reuse=True):
                    object_out_fc_weights = tf.get_variable('weights')
                    object_reg = object_reg + tf.nn.l2_loss(object_out_fc_weights)
                with tf.variable_scope('object_classification', reuse=True):
                    object_classification_weights = tf.get_variable('weights')
                    object_reg = object_reg + tf.nn.l2_loss(object_classification_weights)
                with tf.variable_scope('object_post_fc', reuse=True):
                    object_post_fc_weights = tf.get_variable('weights')
                    object_regCost = object_reg + tf.nn.l2_loss(object_post_fc_weights)

                with tf.variable_scope('base2object', reuse=True):
                    weights = tf.get_variable('weights')
                    object_regCost = object_regCost + tf.nn.l2_loss(weights )
                with tf.variable_scope('V_I_emb', reuse=True):
                    weights = tf.get_variable('weights')
                    object_regCost = object_regCost + tf.nn.l2_loss(weights )
                with tf.variable_scope('vid2obj', reuse=True):
                    weights = tf.get_variable('weights')
                    object_regCost = object_regCost + tf.nn.l2_loss(weights)


                base_C=tf.nn.l2_loss(base_post_fc_weights) +tf.nn.l2_loss(base_action_fc2_weights )+tf.nn.l2_loss(base_class_weights)

                ##########################################################################################################
                if self.hard_label:
                    self.Y_placeholder = tf.placeholder(tf.int64, shape=(None), name='bacth_label')
                    self.Y_placeholder_hot = tf.one_hot(self.Y_placeholder, self.length)
                    weight_per_label = tf.matmul(self.Y_placeholder_hot, tf.cast(tf.expand_dims(class_weight,-1),tf.float32))  # shape [batch_size,]
                    cost=tf.losses.softmax_cross_entropy(self.Y_placeholder_hot,self.predicted_length,weights=tf.squeeze(weight_per_label))
                    correct_pred = tf.equal(tf.argmax(self.predicted_length, 1), tf.argmax(self.Y_placeholder_hot, 1))
                    self.dist = tf.abs(tf.argmax(self.predicted_length, 1) - tf.argmax(self.Y_placeholder_hot, 1))


                else:
                    self.Y_placeholder = tf.placeholder(tf.float32, shape=(None,self.length), name='bacth_label')
                    weight_per_label = tf.matmul(self.Y_placeholder, tf.cast(tf.expand_dims(class_weight,-1),tf.float32))  # shape [batch_size,]
                    cost=tf.losses.softmax_cross_entropy(self.Y_placeholder,self.predicted_length,weights=tf.squeeze(weight_per_label))
                    correct_pred = tf.equal(tf.argmax(self.predicted_length, 1), tf.argmax(self.Y_placeholder, 1))
                    self.dist=tf.abs(tf.argmax(self.predicted_length, 1)-tf.argmax(self.Y_placeholder, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                ##########################################################################################################
                weight_per_label = tf.matmul(self.Xa_placeholder_hot,tf.cast(tf.expand_dims(base_class_weight, -1), tf.float32))  # shape [batch_size,]
                base_cost = tf.losses.softmax_cross_entropy(self.Xa_placeholder_hot, self.base_out,weights=tf.squeeze(weight_per_label))
                base_correct_pred = tf.equal(tf.argmax(self.base_out, 1), tf.argmax(self.Xa_placeholder_hot, 1))
                self.base_accuracy = tf.reduce_mean(tf.cast(base_correct_pred, tf.float32))
                ##########################################################################################################
                ##########################################################################################################
                weight_per_label = tf.matmul(self.Xo_placeholder_hot,tf.cast(tf.expand_dims(object_class_weight, -1), tf.float32))  # shape [batch_size,]
                obj_cost = tf.losses.softmax_cross_entropy(self.Xo_placeholder_hot, self.object_out,weights=tf.squeeze(weight_per_label))
                object_correct_pred = tf.equal(tf.argmax(self.object_out, 1), tf.argmax(self.Xo_placeholder_hot, 1))
                self.object_accuracy = tf.reduce_mean(tf.cast(object_correct_pred, tf.float32))
                ##########################################################################################################

                with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
                    self.lr = tf.placeholder(tf.float32)
                    R=tf.nn.l2_loss(pre_fc_weights) + tf.nn.l2_loss(post_fc_weights)+ tf.nn.l2_loss(action_fc_weights)+ tf.nn.l2_loss(segment_fc_weights)+ tf.nn.l2_loss(length_class_weights)+ tf.nn.l2_loss(segment_fc_weights1)+ tf.nn.l2_loss(action_fc_weights1)+C+base_C
                    self.cost = cost+obj_cost+base_cost+ regul * (R+object_regCost)
                    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)



            # if (First_Time):
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            a_lr=lr
            for n_ep, (data_per_eopch) in enumerate(epoch_gen(training_pack, self.batch_size, num_epochs)):

                loss_per_epoch = 0
                total_items=0
                total_sum = 0
                base_total_sum = 0
                object_total_sum = 0
                total_proximity1 = 0
                total_proximity2 = 0




                for n_batch, (x_f,x_a, y,N,x_48a,x_o,v_i,v_o) in enumerate(data_per_eopch):

                    _, loss_values, acc, length_prob,distance,base_acc,base_prob,obj_prob,obj_acc= sess.run([self.optimizer, self.cost, self.accuracy, self.out_probability,self.dist,self.base_accuracy, self.base_out_probability,
                                                                                                     self.object_out_probability,self.object_accuracy],
                                                                feed_dict={self.Xf_placeholder: x_f,self.Xa_placeholder: x_a, self.Y_placeholder: y,self.NSeg_placeholder:N,self.V_I_placeholder_hot:v_i,self.V_O_placeholder_hot:v_o,
                                                                           self.Xo_placeholder_hot:x_o,self.X48a_placeholder:x_48a,self.training: True, self.keep_p: keep_p,self.lr:a_lr})

                    loss_per_epoch = loss_values + loss_per_epoch
                    total_items=total_items+y.shape[0]
                    total_sum = (acc * y.shape[0]) + total_sum
                    base_total_sum = (base_acc * y.shape[0]) + base_total_sum
                    object_total_sum = (obj_acc * y.shape[0]) + object_total_sum
                    proximity=[]
                    proximity1,proximity2=metrics.proximity_accuracy(proximity,self.length,distance,y,self.hard_label) # if error within 2*sigma then consider correct
                    total_proximity1=total_proximity1+np.sum(proximity1)
                    total_proximity2 = total_proximity2 + np.sum(proximity2)

                if n_ep % 2 == 0:
                    assert total_items==labels_tr.shape[0]
                    val_loss_values, val_acc, val_length_prob,distance_val,base_acc_val,base_prob_val,obj_prob_val,obj_acc_val = sess.run([self.cost, self.accuracy, self.out_probability,self.dist,self.base_accuracy,
                                                                                                    self.base_out_probability,self.object_out_probability,self.object_accuracy],
                                                                                    feed_dict={self.Xf_placeholder: data_f_val,self.Xa_placeholder: data_a_val, self.Y_placeholder:labels_val,self.NSeg_placeholder:nSeg_val,self.V_I_placeholder_hot:V_A_val,
                                                                                               self.V_O_placeholder_hot:V_O_val ,self.Xo_placeholder_hot:object_lb_val,self.X48a_placeholder:_48action_lb_val, self.training: False,self.keep_p: 1.0})
                    p=[]
                    proximity_val1,proximity_val2 = metrics.proximity_accuracy(p, self.length, distance_val,labels_val,self.hard_label)
                    print("Training|| Epoch " + str(n_ep) + " :" + "Loss is :" + str(loss_per_epoch) + " and accuracy is: " + str(total_sum / labels_tr.shape[0])+
                          " and prox#1 is "+ str(total_proximity1/ labels_tr.shape[0])+" and prox#2 is "+ str(total_proximity2/ labels_tr.shape[0]))

                    print("Val. || Epoch " + str(n_ep) + " :" + "Loss is :" + str(val_loss_values) + " and accuracy is: " + str(val_acc) +" and prox#1 is "+ str(np.average(proximity_val1))+" and prox#2 is "+ str(np.average(proximity_val2)))
                    q = np.argmax(val_length_prob, 1)
                    qh = np.bincount(q)
                    pc_acc = metrics.per_class_acc(labels_val, distance_val, self.length, self.hard_label)
                    # print("val set per class accuracy")
                    # print(pc_acc)
                    # print("                                              ")
                    print("              -----------------               ")

                    print("Base: Training|| Epoch " + str(n_ep) + " :" + " and accuracy is: " + str(base_total_sum /V_A.shape[0]))

                    print("Base: Val. || Epoch " + str(n_ep) + " :"  + " and accuracy is: " + str(base_acc_val))
                    q = np.argmax(base_prob_val, 1)
                    qh = np.bincount(q)
                    print("                                        ")
                    metrics.print2file_accuracy_for_one_hot_output(object_total_sum /data_a_tr.shape[0],obj_prob,n_ep,training=True,nclass=self.nObjects,true=x_o,filename='object_accuracy_training.txt')
                    metrics.print2file_accuracy_for_one_hot_output(obj_acc_val, obj_prob_val, n_ep, training=False,nclass=self.nObjects,true=object_lb_val,filename='object_accuracy_test.txt')

                    print("                                        ")
                    print("              * * * * * *               ")
                    print("#############################################")

                if n_ep %  (num_epochs-1) == 0 and n_ep!=0:
                    # print("fake saving")
                    save_variables(sess, var_save_path)

            sess.close()

    def predict(self,sess,x_f_test,x_a_test,nSeg_test,x_a_v,x_o_v):



        length_probability,object_probability,base_prob=sess.run([self.out_probability,self.object_out_probability,self.base_out_probability],feed_dict={self.Xf_placeholder: x_f_test,self.Xa_placeholder: x_a_test,self.NSeg_placeholder:nSeg_test,
                                                                           self.V_I_placeholder_hot:x_a_v,self.V_O_placeholder_hot:x_o_v,
                                                                           self.training: False,self.keep_p: 1.0})

        return length_probability,object_probability,base_prob
