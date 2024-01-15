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
    
    def __init__(self):
        self.G = []
        self.type = 'RandomWalk'

    def train(self, G, paths):
        '''
        Takes a maritime traffic network graph as input
        '''
        for u, v, data in G.edges(data=True):
            data['weight'] = 0
            del G.edges[u, v]['geometry']
            del G.edges[u, v]['inverse_weight']
            del G.edges[u, v]['length']
            del G.edges[u, v]['direction']

        for path in paths:
            for i  in range(0, len(path)-1):
                u = path[i]
                v = path[i+1]
                G[u][v]['weight'] += 1
        
        self.G = G

    
    def sample_paths(self, start_node, n_walks, n_steps, method='weighted'):
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
        Docstring
        '''
        result_list=[]
        for index, row in paths.iterrows():
            mmsi = row['mmsi']
            path = row['path']
            start_node = path[0:n_start_nodes]
            
            predictions = self.sample_paths(start_node, n_walks, n_steps, method)
            predictions = dict(heapq.nlargest(np.min([len(predictions), n_predictions]), predictions.items(), key=lambda x: x[1]))
            for key, value in predictions.items():
                predicted_path = [x for x in key]
                result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                    'prediction': tuple(predicted_path), 'probability':value/n_walks})
                    
        predictions = pd.DataFrame(result_list)
        return predictions

        
        
            