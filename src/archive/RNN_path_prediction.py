from RNN_TrajModel.main import *
from RNN_TrajModel.geo import Map, GeoPoint
from RNN_TrajModel.ngram_model import N_gram_model
from RNN_TrajModel.trajmodel import TrajModel
import time
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import warnings
warnings.filterwarnings('ignore')


class RNNPathPrediction:
    
    def __init__(self):
        self.model = None
        self.config = None
        self.type = 'RNN'
        self.task = None
        self.train_metrics = {'max_prediction_acc': [], 'loss': [], 'loss_p': []}
        self.test_metrics = {'max_prediction_acc': [], 'loss': [], 'loss_p': []}
        self.valid_metrics = {'max_prediction_acc': [], 'loss': [], 'loss_p': []}
        self.path_to_model = None
        self.saver = None

    def train(self, config_file, path_to_model):
        config = Config(config_file)
        self.config = config
        # set log file
        timestr = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())) # use for naming the log file
        if config.direct_stdout_to_file:
            model_str = config.model_type
            config.log_filename = "log_" + timestr + "_" + config.dataset_name + "_" + model_str + ".txt"
            config.log_file = open(config.log_filename, 'w')
            sys.stdout = config.log_file

        # process data
        routes, train, valid, test = read_data(config.dataset_path, config.data_size, config.max_seq_len)
        print("successfully read %d routes" % sum([len(train), len(valid), len(test)]))
        max_edge_id = max([max(route) for route in routes])
        min_edge_id = min([max(route) for route in routes])
        print("min_edge_id = %d, max_edge_id = %d" % (min_edge_id, max_edge_id))
        max_route_len = max([len(route) for route in routes])
        route_lens = [len(route) for route in routes]
        print("train:%d, valid:%d, test:%d" % (len(train), len(valid), len(test)))
        print(max(route_lens))

        def count_trans(roadnet, data):
            # initialization
            print("start initialization")
            trans = []
            for edge in roadnet.edges:
                adjs = {}
                for adj_edge_id in edge.adjList_ids:
                    adjs[adj_edge_id] = 0
                trans.append(adjs)

            # do stats
            print("start stats")
            for route in data:
                for i in range(len(route) - 1):
                    trans[route[i]][route[i + 1]] += 1

            f = open("count_trans", "w")
            for edge in roadnet.edges:
                f.write(str(edge.id) + " ")
                for adj_edge_id in edge.adjList_ids:
                    f.write("|" + str(adj_edge_id) + " :\t" + str(trans[edge.id][adj_edge_id]) + "\t")
                f.write("\n")
            f.close()

        # load map
        GeoPoint.AREA_LAT = 70 # the latitude of the testing area. In fact, any value is ok in this problem.
        roadnet = Map()
        roadnet.open(config.map_path)

        # set config
        config.set_config(routes, roadnet)
        config.printf()

        # extract map info
        mapInfo = MapInfo(roadnet, config)

        if config.eval_ngram_model:
            # n-gram model eval
            markov_model = N_gram_model(roadnet, config)
            # markov_model.train_and_eval(train, valid, 5, config.max_seq_len, given_dest=True,use_fast=True, compute_all_gram=True)
            print("======================test set========================")
            markov_model.train_and_eval_given_dest(train, test, 3, 600, use_fast=True)
            print("======================valid set========================")
            markov_model.train_and_eval_given_dest(train, valid, 3, 600, use_fast=True)
            input()
        
        # construct model
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            model_scope = "Model"
            with tf.name_scope("Train"):
                with tf.variable_scope(model_scope, reuse=None, initializer=initializer):
                    model = TrajModel(not config.trace_hid_layer, config, train, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

            with tf.name_scope("Valid"):
                with tf.variable_scope(model_scope, reuse=True):
                    model_valid = TrajModel(False, config, valid, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

            with tf.name_scope("Test"):
                with tf.variable_scope(model_scope, reuse=True):
                    model_test = TrajModel(False, config, test, model_scope=model_scope, map=roadnet, mapInfo=mapInfo)

            # sv = tf.train.Supervisor(logdir=config.load_path)
            # with sv.managed_session() as sess:
            sess_config = tf.ConfigProto()
            # initialize saver to save model
            saver = tf.train.Saver()
            # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
            # sess_config.gpu_options.allow_growth = True
            with tf.Session(config=sess_config) as sess:
                # stuff for ckpt
                ckpt_path = None
                if config.load_ckpt:
                    print('Input training ckpt filename (at %s): ' % config.load_path)
                    if PY3:
                        ckpt_name = input()
                    else:
                        ckpt_name = raw_input()
                    print(ckpt_name)
                    ckpt_path = os.path.join(config.load_path, ckpt_name)
                    print('try loading ' + ckpt_path)
                if ckpt_path and tf.gfile.Exists(ckpt_path):
                    print("restoring model trainable params from %s" % ckpt_path)
                    model.saver.restore(sess, ckpt_path)
                else:
                    if config.load_ckpt:
                        print("restore model params failed")
                    print("initialize all variables...")
                    sess.run(tf.initialize_all_variables())

                # benchmark for testing speed
                print("speed benchmark for get_batch()...")
                how_many_tests = 1000
                t1 = time.time()
                for _ in range(how_many_tests):
                    model.get_batch(model.data, config.batch_size, config.max_seq_len)
                t2 = time.time()
                print("%.4f ms per batch, %.4fms per sample, batch_size = %d" % (float(t2-t1)/how_many_tests*1000.0,
                                                                                 float(t2-t1)/how_many_tests/config.batch_size*1000.0,
                                                                                 config.batch_size))

                # use for timeline trace (unstable, need lots of memory)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                print('start benchmarking...')
                model.speed_benchmark(sess, config.samples_for_benchmark)
                # timeline generation
                model_valid.speed_benchmark(sess, config.samples_for_benchmark)
                print("start training...")

                if config.direct_stdout_to_file:
                    config.log_file.close()
                    config.log_file = open(config.log_filename, "a+")
                    sys.stdout = config.log_file

                # let's go :)
                for ep in range(config.epoch_count):
                    print('Epoch ', ep)
                    if not config.eval_mode:
                        model.train_epoch(sess, train)
                    #cumulative_losses_train = model.eval(sess, train, True, True, model_train=model)
                    cumulative_losses_valid = model_valid.eval(sess, valid, True, True, model_train=model)
                    cumulative_losses_test = model_test.eval(sess, test, False, False, model_train=model)
                    print('_________________________________________________')
                    #for key, v in cumulative_losses_train.items():
                    #    self.train_metrics[key].append(v)
                    for key, v in cumulative_losses_test.items():
                        self.test_metrics[key].append(v)
                    for key, v in cumulative_losses_valid.items():
                        self.valid_metrics[key].append(v)
                # save model
                saver.save(sess, path_to_model)
                
        #input()
        self.model = model
        self.saver = saver
        self.path_to_model = path_to_model

    def plot_train_test_metrics(self, test_val_only=True):
        '''
        Plot metrics for training, test and validation set for each epoch
        '''
        # Create subplots
        num_keys = len(self.test_metrics)
        fig, axes = plt.subplots(ncols=3, figsize=(8, 3))
        
        # Loop through each key and create a subplot
        for i, key in enumerate(self.test_metrics.keys()):
            if test_val_only==False:
                train_values = self.train_metrics[key]
            test_values = self.test_metrics[key]
            valid_values = self.valid_metrics[key]
            x = np.arange(len(test_values))
            
            # get the axis to plot on
            ax = axes[i]
            
            # plot
            if test_val_only==False:
                ax.plot(x, train_values, label='Train', color='b')
            ax.plot(x, test_values, label='Test', color='r')
            ax.plot(x, valid_values, label='Test', color='g')

            # set axis limits
            if key in ['max_prediction_acc']:
                ax.set_ylim([0,1])
            
            # add labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.legend()
        
        plt.tight_layout()
        plt.show()