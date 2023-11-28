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
    
    def __init__(self):
        self.G = []
        self.type = 'Dijkstra'

    def train(self, G, paths):
        '''
        Takes a maritime traffic network and a selection of trajectory paths through that network as input
        Computes a weight for each edge, based on how many ships travelled along the edge
        '''
        # reset any previously calculated edge weights
        for u, v, data in G.edges(data=True):
            data['weight'] = 0
            data['inverse_weight'] = 0
        
        for path in paths:
            for i  in range(0, len(path)-1):
                u = path[i]
                v = path[i+1]
                G[u][v]['weight'] += 1
        
        for u, v, in G.edges():
            if G[u][v]['weight'] > 0:
                G[u][v]['inverse_weight'] = 1/G[u][v]['weight'] * G[u][v]['length']
            else:
                G[u][v]['inverse_weight'] = np.inf
        
        self.G = G

    def predict(self, orig, dest):
        '''
        outputs the shortest path in the network using Dijkstra's algorithm.
        :param orig: int, ID of the start waypoint
        :param dest: int, ID of the destination waypoint
        '''     
        try:
            # compute shortest path using dijsktra's algorithm (outputs a list of nodes)
            shortest_path = nx.dijkstra_path(self.G, orig, dest, weight='inverse_weight')
            return shortest_path
        except:
            print(f'Nodes {orig} and {dest} are not connected. Exiting...')
            return []
        
            