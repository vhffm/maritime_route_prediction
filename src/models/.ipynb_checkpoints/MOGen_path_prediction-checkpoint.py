import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import pathpy as pp
import numpy as np
from datetime import timedelta, datetime
from shapely.geometry import Point, LineString, MultiLineString
from shapely import ops
import matplotlib.pyplot as plt
import heapq
import folium
import time
import warnings
import pickle
import sys

# add paths for modules
sys.path.append('../features')

# import modules
import geometry_utils

class MOGenPathPrediction:
    '''
    A model for route prediction with or without target information.
    At the core, the model uses the MOGen path prediction model (pathpy project: https://github.com/pathpy/pathpy).
    '''
    def __init__(self):
        self.model = []
        self.type = 'MOGen'
        self.optimal_order = 0
        self.order = self.optimal_order

    def train(self, paths, max_order, model_selection=True, training_mode='partial'):
        '''
        Trains a MOGen model based on observed paths in a network
        ====================================
        Params:
        paths: (list of lists) List of paths, for example: [[2, 3, 4], [1, 7, 9]]
        max_order: (int) The maximum order of the MOGen model. 
                   If we want to make predictions specifying a sequence of n start nodes, max_order needs to be at least n.
        model_selection: (bool) If True, then MOGen trains a model up to max_order, including all models < max_order
                                If False, only a model with max_order is trained
        training_mode: 'full' - model is trained on input paths and sub-paths, e.g. when an input path is [1, 2, 3, 4], the model is trained on
                                [1, 2, 3, 4], [2, 3, 4], [3, 4]
                       'partial' - model is trained only on actual input paths
        ====================================    
        '''
        # create a pathpy path collection and add all training paths to it
        pc = pp.PathCollection()
        for raw_path in paths:
            # train model on path and all sub-paths
            if training_mode == 'full':
                for j in range(0, len(raw_path)-1):
                    sub_path = raw_path[j:]
                    str_path = [str(i) for i in sub_path]  # convert node IDs to strings
                    path = pp.Path(*str_path)
                    pc.add(path)
            # train model on original paths only
            else:
                str_path = [str(i) for i in raw_path]  # convert node IDs to strings
                path = pp.Path(*str_path)
                pc.add(path)
    
        # initialize MOGen Model
        # creates MOGen models up to order max_order
        # when model_selection=False, only a model of order max_order is created
        model = pp.MOGen(paths=pc, max_order=max_order, model_selection=model_selection)
    
        # find best MOGen model
        model.fit()

        self.optimal_order = model.optimal_maximum_order
        self.order = self.optimal_order
        self.model = model

    def sample_paths(self, start_node, n_paths, G, seed=42, verbose=True):
        '''
        Samples a certain number of paths given a start node
        ====================================
        Params:
        start_node: (list of int) Single start node or start node sequence, for example: [1], or [1, 2, 3]
                                  Sequence cannot be longer than max_order of the MOGen model
        n_paths: (int) number of paths to sample
        ====================================
        Returns:
        normalized_predictions_dict: (dict) Dictionary of paths and their probability of occurence, sorted by probability (descending)
        '''
        if len(start_node) > self.model.max_order:
            red_background = "\033[48;2;255;200;200m"
            reset_style = "\033[0m"
            print(f'{red_background}Error: Length of the start node sequence ({len(start_node)}) is too long. Maximum order of the prediction model is {self.model.max_order}.')
            print(f'Reduce length of start node sequence or train a higher order model.{reset_style}')
            return
        
        # convert start node format from int list to string tuple (input format for MOGen model)
        start_node_str = tuple(str(node) for node in start_node)
        
        # make predictions based on start node
        try:
            predictions = self.model.predict(no_of_paths=n_paths, max_order=self.order, 
                                             start_node=start_node_str, seed=seed, verbose=verbose, paths_per_process=100)
        except:
            predictions = self.model.predict(no_of_paths=n_paths, max_order=1, 
                                             start_node=start_node_str, seed=seed, verbose=verbose, paths_per_process=100)
        # sort predictions by frequency of occurrence
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        normalized_predictions_dict={}
        for key, val in sorted_predictions.items():
            node_sequence = tuple(int(x) for x in key)
            normalized_predictions_dict[node_sequence] = val/n_paths  # probability of occurrence

        return normalized_predictions_dict

    def predict_next_nodes(self, start_node, G, n_steps=1, n_predictions=1, n_walks=100, verbose=True):
        '''
        Given a start node or a start node sequence, the model predicts the next node(s) (route prediction without target information)
        ====================================
        Params:
        start_node: (list of int) Single start node or start node sequence, for example: [1], or [1, 2, 3]
                                  Sequence cannot be longer than max_order of the MOGen model
        G: (networkx graph object) the graph underlying the traffic network
        n_steps: (int) the prediction horizon (if -1, we get path predictions of arbitrary length)
        n_predictions: (int) number of node candidates to predict
        n_walks: the number of random walks performed by the MOGen model. The higher this number, the better the prediction of next node(s)
        verbose: if True, program flow is printed on screen 
        ====================================
        Returns:
        sorted_predictions: dictionary of node sequences and their predicted probabilities sorted by the latter
        '''
        # predict n_walks path from start_node
        predicted_paths = self.sample_paths(start_node, n_walks, G, verbose=verbose)
        n_start_nodes = len(start_node)
        sums_dict = {}
        for key, val in predicted_paths.items():
            if n_steps > 0:
                node_sequence = key[0:n_start_nodes+n_steps]
            else:
                node_sequence = key
            # check if the predicted path is valid on the graph G
            if geometry_utils.is_valid_path(G, [x for x in node_sequence]):
                # if the path is valid, either append it to the dictionary of predictions...
                if node_sequence not in sums_dict:
                    if len(node_sequence) >= n_start_nodes+n_steps:
                        sums_dict[node_sequence] = val
                # ... or increase its probability
                else:
                    sums_dict[node_sequence] += val
        # only retain the desired number of predicted alternatives
        if n_predictions == -1:
            predictions = heapq.nlargest(len(sums_dict), sums_dict.items(), key=lambda x: x[1])
        else:
            predictions = heapq.nlargest(np.min([len(sums_dict), n_predictions]), sums_dict.items(), key=lambda x: x[1])
        
        # convert to dictionary and sort
        predictions = dict(predictions)
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
         
        return sorted_predictions

    def predict_path(self, start_node, end_node, G, n_predictions=1, n_walks=100, verbose=True):
        '''
        Given a start node or a start node sequence and an end node, the model predicts likely paths between these
        (route prediction with target information)
        ====================================
        Params:
        start_node: (list of int) node ID(s) of single start node or start node sequence, for example: [1], or [1, 2, 3]
                    Sequence cannot be longer than max_order of the MOGen model
        end_node: (int) node ID of end node, e.g. 5
        G: (networkx graph object) the graph underlying the traffic network
        n_predictions: (int) number of path candidates to predict
        n_walks: (int) the number of random walks performed by the MOGen model. The higher this number, the better the prediction
        verbose: if True, program flow is printed on screen
        ====================================
        Returns:
        normalized_sorted_predictions: dictionary of paths and their predicted probabilities
        flag: If True, a path from start to destination has been found.
        '''
        # predict n_walks paths from start_node
        predicted_paths = self.sample_paths(start_node, n_walks, G, verbose=verbose)
        sums_dict, flag = self.return_valid_paths(predicted_paths, start_node, end_node, G, n_walks)
        # if no path was found, retry with more random walks
        while (n_walks < 4000) & (flag == False):
            n_walks = n_walks*2
            #print(f'No path was found. Retrying with more random walks {n_walks}')
            predicted_paths = self.sample_paths(start_node, n_walks, G, verbose=verbose)
            sums_dict, flag = self.return_valid_paths(predicted_paths, start_node, end_node, G, n_walks)
        # only retain the desired number of predicted alternatives
        if n_predictions == -1:
            predictions = heapq.nlargest(len(sums_dict), sums_dict.items(), key=lambda x: x[1])
        else:
            predictions = heapq.nlargest(np.min([len(sums_dict), n_predictions]), sums_dict.items(), key=lambda x: x[1])
        
        # convert to dictionary and sort
        predictions = dict(predictions)
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        # normalize observed predictions, to get probabilities
        total_sum = sum(sums_dict.values())
        normalized_sorted_predictions = {key: value / total_sum for key, value in sorted_predictions.items()}

        return normalized_sorted_predictions, flag

    def return_valid_paths(self, predicted_paths, start_node, end_node, G, n_walks):
        '''
        Given a start and end node and a set of predicted paths, this method returns only the paths that contain the start and end node
        ====================================
        Params:
        predicted_paths: (dict) a dictionary of path predictions
        start_node: (list of int) node ID(s) of single start node or start node sequence, for example: [1], or [1, 2, 3]
                    Sequence cannot be longer than max_order of the MOGen model
        end_node: (int) node ID of end node, e.g. 5
        G: (networkx graph object) the graph underlying the traffic network
        n_walks: (int) the number of random walks performed by the MOGen model. The higher this number, the better the prediction
        ====================================
        Returns:
        sums_dict: (dict) dictionary of paths and their associated probabilities
        flag: If True, at least one path from start to destination has been found.
        '''
        sums_dict = {}
        flag = False
        for key, val in predicted_paths.items():
            node_sequence = key
            # check if the predicted path is valid on the graph G
            is_valid_path = geometry_utils.is_valid_path(G, [x for x in node_sequence])
            # check if the predicted path is valid and contains the end node
            if (is_valid_path) & (end_node in node_sequence):
                #print('Success! Found a path to the end node.')
                flag = True
                # clip node sequence to end at the specified end_node
                index_to_clip = node_sequence.index(end_node)
                clipped_node_sequence = node_sequence[:index_to_clip+1]
                # either append it to the dictionary of predictions...
                if clipped_node_sequence not in sums_dict:
                    sums_dict[clipped_node_sequence] = val*n_walks
                # ... or increase its probability
                else:
                    sums_dict[clipped_node_sequence] += val*n_walks
        return sums_dict, flag                            

    def predict(self, prediction_task, paths, G, n_start_nodes=1, n_steps=1, n_predictions=1, n_walks=100, order=0):
        '''
        Method for inference. Given an observed path, predict the entire route from start to destination (if destination information is available).
        ====================================
        Params:
        prediction_task: (string) 'next_nodes' for route prediction without destination information
                                  'path' for route prediction with destination information
        paths: (Dataframe) the ground truth paths and the vessel mmsi
               if the entire ground truth path is unknown, specify only the start nodes and optionally the end node, e.g.
               mmsi     path
               12345    [5, 30]   (where 5 is the ID of the start node and 30 the ID of the end node)
        G: (networkx graph object) the graph underlying the traffic network
        n_start_nodes: (int) number of observed nodes in the path prefix
        n_steps: (int) prediction horizon
        n_predictions: (int) number of output predictions. E.g. n_predictions = 3 yields the top 3 predictions based on frequency of occurrence
        n_walks: (int) number of random walks for sampling
        order: (int) force the order of the MOGen model for prediction. If 0, the model chooses the optimal order automatically (recommended).
        ====================================
        Returns:
        predictions: (Dataframe) the predicted paths
        '''
        result_list=[]

        # set order of MOGen model. When order=0, we set order to the optimal order determined by MOGen
        if order > 0:
            self.order = order
        
        print(f'Making predictions for {len(paths)} samples with MOGen of order {self.order}')
        print(f'Progress:', end=' ', flush=True)
        count = 0  # initialize a counter that keeps track of the progress
        percentage = 0  # percentage of progress
        
        for index, row in paths.iterrows():
            mmsi = row['mmsi']
            path = row['path']
            start_node = path[0:n_start_nodes]
            end_node = path[-1]
           
            if prediction_task == 'path':
                try:
                    prediction, flag = self.predict_path(start_node, end_node, G, 
                                                         n_predictions=n_predictions, n_walks=n_walks, verbose=False)
                    if flag:
                        for key, value in prediction.items():
                            predicted_path = [x for x in key]
                            result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                                'prediction': tuple(predicted_path), 'probability':value})
                    else:
                        result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                            'prediction': [], 'probability':np.nan})
                except:
                    result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                        'prediction': [], 'probability':np.nan})
            
            elif prediction_task == 'next_nodes':
                try:
                    prediction = self.predict_next_nodes(start_node, G, n_steps=n_steps, 
                                                         n_predictions=n_predictions, n_walks=n_walks, verbose=False)
                    for key, value in prediction.items():
                        predicted_path = [x for x in key]
                        result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                            'prediction': tuple(predicted_path), 'probability':value})
                except:
                    result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                        'prediction': [], 'probability':np.nan})   
            
            else:
                print('invalid prediction task')
            
            # report progress
            count += 1
            if count/len(paths) > 0.1:
                count = 0
                percentage += 10
                print(f'{percentage}%...', end='', flush=True)
                    
        print('Done!')
        
        predictions = pd.DataFrame(result_list)
        return predictions
        