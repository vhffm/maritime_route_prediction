import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
from datetime import timedelta, datetime
import networkx as nx
import matplotlib.pyplot as plt
import time
import warnings
import sys

# add paths for modules
sys.path.append('../features')

# import modules
import geometry_utils


class DijkstraPathPrediction:
    '''
    A model for route prediction with target information.
    The model uses Dijkstra's algorithm for path computation.
    '''
    def __init__(self):
        self.G = []
        self.type = 'Dijkstra'

    def train(self, G, paths):
        '''
        This method trains the prediction model by computing edge weights from observed paths
        The edge weights are based on edge length and the number of passages along each edge.
        Two additional edge weights in addition to the edge length are computed:
            - inverse_passages: The inverse of the number of registered passages along an edge
            - inverse_density: The inverse traffic density along an edge. 
                               The traffic density is computed as the fraction of passages and edge length (passages per meter)
        The reweighted graph is saved as an attribute.
        ====================================
        Params:
        G: (networkx graph) The maritime traffic network to make route predictions on
        paths: (list of int) a list of paths for learning edge weights
        ====================================
        Returns
        '''
        # reset any previously calculated edge weights
        for u, v, data in G.edges(data=True):
            data['weight'] = 0
        
        # compute the number of passages along each edge from observed paths
        for path in paths:
            for i  in range(0, len(path)-1):
                u = path[i]
                v = path[i+1]
                G[u][v]['weight'] += 1
        
        # compute edge weights inverse_passages and inverse_density
        for u, v, in G.edges():
            if G[u][v]['weight'] > 0:
                G[u][v]['inverse_density'] = 1/G[u][v]['weight'] * G[u][v]['length']
                G[u][v]['inverse_passages'] = 1/G[u][v]['weight']
            else:
                G[u][v]['inverse_density'] = np.inf
                G[u][v]['inverse_passages'] = np.inf
        
        self.G = G

    
    def predict_path(self, orig, dest, weight='inverse_density'):
        '''
        Computes the shortest path on the graph between two nodes using Dijkstra's algorithm.
        ====================================
        Params:
        orig: (int) ID of the start waypoint
        dest: (int) ID of the destination waypoint
        weight: (string) the edge weight for Dijkstra's algorithm ('length', 'inverse_passages' or 'inverse_density')
        ====================================
        Returns:
        shortest_path: (list) shortest path found by Dijkstra's algorithm
        flag: True if path was found. False if no path exists between the two nodes.
        '''     
        try:
            # compute shortest path using dijsktra's algorithm (outputs a list of nodes)
            shortest_path = nx.dijkstra_path(self.G, orig, dest, weight=weight)
            return shortest_path, True
        except:
            print(f'Nodes {orig} and {dest} are not connected. Exiting...')
            return [], False

    
    def predict(self, paths, n_start_nodes=1, weight='inverse_density'):
        '''
        Method for inference. Given an observed path, predict the entire route from start to destination.
        ====================================
        Params:
        paths: (Dataframe) the ground truth paths and the vessel mmsi
               if the entire ground truth path is unknown, specify only the start nodes and end node, e.g.
               mmsi     path
               12345    [5, 30]   (where 5 is the ID of the start node and 30 the ID of the end node)
        n_start_nodes: (int) number of observed nodes in the path prefix (needs to be shorter than the length of the specified ground truth paths)
        weight: (string) the edge weight for Dijkstra's algorithm ('length', 'inverse_passages' or 'inverse_density')
        ====================================
        Returns:
        predictions: (Dataframe) the predicted paths
        '''
        result_list=[]
        
        print(f'Making predictions for {len(paths)} samples')
        print(f'Progress:', end=' ', flush=True)
        count = 0  # initialize a counter that keeps track of the progress
        percentage = 0  # percentage of progress
        
        for index, row in paths.iterrows():
            mmsi = row['mmsi']
            path = row['path']
            start_node = path[0:n_start_nodes]
            end_node = path[-1]
            # predict path
            predicted_path, flag = self.predict_path(start_node[-1], end_node, weight)
            predicted_path = start_node[0:-1] + predicted_path  # prepend remainder of start node
            # save results to dict
            if flag:
                result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                    'prediction': tuple(predicted_path), 'probability':1})
            else:
                result_list.append({'mmsi': mmsi, 'ground_truth': tuple(path), 
                                    'prediction': [], 'probability':np.nan})

            # report progress
            count += 1
            if count/len(paths) > 0.1:
                count = 0
                percentage += 10
                print(f'{percentage}%...', end='', flush=True)
                    
        print('Done!')
        
        predictions = pd.DataFrame(result_list)
        return predictions
        
            