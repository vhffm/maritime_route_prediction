import pandas as pd
import numpy as np
import networkx as nx
import random
import heapq
import time
import warnings
import sys

# add paths for modules
sys.path.append('../features')

# import modules
import geometry_utils


class RandomWalkPathPrediction:
    '''
    A statistical model for route prediction without target information.
    The model can take the form of a random predictor or a Markov chain.
    '''
    
    def __init__(self):
        self.G = []
        self.type = 'RandomWalk'

    def train(self, G_in, paths):
        '''
        This method trains the prediction model by computing edge weights from observed paths
        The reweighted graph is saved as an attribute.
        ====================================
        Params:
        G_in: (networkx graph) The maritime traffic network to make route predictions on
        paths: (list of int) a list of paths for learning edge weights
        ====================================
        Returns
        '''
        G = G_in.copy()
        # reset edge features
        for u, v, data in G.edges(data=True):
            data['weight'] = 0
            del G.edges[u, v]['geometry']
            del G.edges[u, v]['inverse_weight']
            del G.edges[u, v]['length']
            del G.edges[u, v]['direction']

        # compute passages along each edge from paths and assign the value as edge weight
        for path in paths:
            for i  in range(0, len(path)-1):
                u = path[i]
                v = path[i+1]
                G[u][v]['weight'] += 1
        
        self.G = G

    
    def sample_paths(self, start_node, n_walks, n_steps, method='weighted'):
        '''
        This method samples paths on the reweighted graph from random walks
        ====================================
        Params:
        start_node: (list of int) the IDs of the observed path prefix
        n_walks: (int) number of random walks for sampling
        n_steps: (int) prediction horizon
        method: (string) either 'weighted' for a Markov chain (uses edge weights) 
                         or 'random' for a random predictor (does not use edge weights)
        ====================================
        Returns:
        sorted_paths: (dict) the predicted paths and their frequency of occurrence
        '''
        path_dict = {}
        for i in range(0, n_walks):
            # random walk
            path = start_node.copy()
            prev_node = start_node[-1]
            for k in range(0, n_steps):
                try:
                    if method=='random':
                        node = random.choice(list(self.G.successors(prev_node)))
                        prev_node = node
                        path.append(node)
                    elif method=='weighted':
                        # Get the successors and their weights
                        successors = list(self.G.successors(prev_node))
                        weights = [self.G[prev_node][succ]['weight'] for succ in successors]
                        # Normalize the weights to probabilities
                        total_weight = sum(weights)
                        probabilities = [weight / total_weight for weight in weights]
                        # Choose the next node based on weights
                        node = random.choices(successors, weights=probabilities)[0]
                        # Update prev_node and append to the path
                        prev_node = node
                        path.append(node)
                    else:
                        print('Choose a valid method: "random" or "weighted".')
                        break
                except:
                    break
            path = tuple(path)
            # append result of random walk to dictionary...
            if path not in path_dict:
                path_dict[path] = 1
            # ... or increase its frequency of occurence
            else:
                path_dict[path] += 1
        # sort predictions by frequency of occurrence
        sorted_paths = dict(sorted(path_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_paths

    
    def predict(self, paths, n_start_nodes=1, n_steps=1, n_predictions=1, n_walks=100, method='weighted'):
        '''
        This method predicts one or multiple paths given an observation
        ====================================
        Params:
        paths: (Dataframe) the ground truth paths and the vessel mmsi
               if the entire ground truth path is unknown, specify the observed path prefix, e.g.
               mmsi     path
               12345    [5, 6]   (where 5 and 6 are the IDs of the observed start nodes)
        n_start_nodes: (int) number of observed nodes in the path prefix (needs to be shorter than the length of the specified ground truth paths)
        n_steps: (int) prediction horizon
        n_predictions: (int) number of output predictions. E.g. n_predictions = 3 yields the top 3 predictions based on frequency of occurrence
        n_walks: (int) number of random walks for sampling
        method: (string) either 'weighted' for a Markov chain (uses edge weights) 
                         or 'random' for a random predictor (does not use edge weights)
        ====================================
        Returns:
        predictions: (Dataframe) the predicted paths and associated probability
        '''
        result_list=[]
        for index, row in paths.iterrows():
            mmsi = row['mmsi']
            path = row['path']
            start_node = path[0:n_start_nodes]
            
            # sample random walks
            predictions = self.sample_paths(start_node, n_walks, n_steps, method)
            # sort predictions by frequency of occurrence
            predictions = dict(heapq.nlargest(np.min([len(predictions), n_predictions]), predictions.items(), key=lambda x: x[1]))
            # write results to dataframe
            for key, value in predictions.items():
                predicted_path = [x for x in key]
                result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                    'prediction': tuple(predicted_path), 'probability':value/n_walks})
                    
        predictions = pd.DataFrame(result_list)
        return predictions           