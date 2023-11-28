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
    
    def __init__(self):
        self.model = []
        self.type = 'MOGen'

    def train(self, paths, max_order, model_selection=True, training_mode='partial'):
        '''
        Trains a MOGen model based on observed paths in a network
        ====================================
        Params:
        paths: List of paths, for example: [[2, 3, 4], [1, 7, 9]]
        max_order: The maximum order of the MOGen model. 
                   If we want to make predictions specifying a sequence of n start nodes, max_order needs to be at least n.
        model_selection: If True, then MOGen trains a model up to max_order, including all models < max_order
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
    
        self.model = model

    def predict_paths(self, start_node, n_paths, G, seed=42, verbose=True):
        '''
        Predicts a certain number of paths from a start node
        ====================================
        Params:
        start_node: Single start node or start node sequence, for example: [1], or [1, 2, 3]
                    Sequence cannot be longer than max_order of the MOGen model
        n_paths: number of paths to predict
        ====================================
        Returns:
        sorted_predictions: Dictionary of paths and their probability of occurence, sorted by probability (descending)
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
        predictions = self.model.predict(no_of_paths=n_paths, max_order=self.model.max_order, start_node=start_node_str, seed=seed, verbose=verbose)
        # sort predictions by frequency of occurrence
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        sorted_predictions_dict={}
        for key, val in sorted_predictions.items():
            node_sequence = tuple(int(x) for x in key)
            sorted_predictions_dict[node_sequence] = val/n_paths  # probability of occurrence
        
        return sorted_predictions_dict

    def predict_next_nodes(self, start_node, G, n_steps=1, n_predictions=1, n_walks=100, verbose=True):
        '''
        Given a start node or a start node sequence, the model predicts the next node(s)
        ====================================
        Params:
        start_node: Single start node or start node sequence, for example: [1], or [1, 2, 3]
                    Sequence cannot be longer than max_order of the MOGen model
        G: the graph underlying the traffic network (networkx graph object)
        n_predictions: number of node candidates to predict
        n_steps: the maximum length of the predicted path ahead (if -1, then we look at path predictions of unlimited length)
        n_walks: the number of random walks performed by the MOGen model. The higher this number, the better the prediction of next node(s)
        ====================================
        Returns:
        predictions: dictionary of nodes sequences and their predicted probabilities
        '''
        # predict n_walks path from start_node
        predicted_paths = self.predict_paths(start_node, n_walks, G, verbose=verbose)
        index = len(start_node)
        sums_dict = {}
        for key, val in predicted_paths.items():
            if n_steps > 0:
                node_sequence = key[index:index+n_steps]
            else:
                node_sequence = key[index:]
            # check if the predicted path is valid on the graph G
            if geometry_utils.is_valid_path(G, start_node + [x for x in node_sequence]):
                # if the path is valid, either append it to the dictionary of predictions...
                if node_sequence not in sums_dict:
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
        ====================================
        Params:
        start_node: node ID(s) of single start node or start node sequence, for example: [1], or [1, 2, 3]
                    Sequence cannot be longer than max_order of the MOGen model
        end_node: node ID of end node, e.g. 5
        G: the graph underlying the traffic network (networkx graph object)
        n_predictions: number of node candidates to predict
        n_walks: the number of random walks performed by the MOGen model. The higher this number, the better the prediction of next node(s)
        ====================================
        Returns:
        predictions: dictionary of paths and their predicted probabilities
        '''
        # predict n_walks path from start_node
        predicted_paths = self.predict_paths(start_node, n_walks, G, verbose=verbose)
        sums_dict, flag = self.return_valid_paths(predicted_paths, start_node, end_node, G, n_walks)
        while (n_walks < 4000) & (flag == False):
            n_walks = n_walks*2
            #print(f'No path was found. Retrying with more random walks {n_walks}')
            predicted_paths = self.predict_paths(start_node, n_walks, G, verbose=verbose)
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
        total_sum = sum(sorted_predictions.values())
        normalized_sorted_predictions = {key: value / total_sum for key, value in sorted_predictions.items()}

        return normalized_sorted_predictions, flag

    def return_valid_paths(self, predicted_paths, start_node, end_node, G, n_walks):
        sums_dict = {}
        index = len(start_node)
        flag = False
        for key, val in predicted_paths.items():
            node_sequence = key[index:]
            # check if the predicted path is valid on the graph G
            is_valid_path = geometry_utils.is_valid_path(G, start_node+[x for x in node_sequence])
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


        