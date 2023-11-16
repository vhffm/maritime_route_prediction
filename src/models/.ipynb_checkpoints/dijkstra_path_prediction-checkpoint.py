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

    def evaluate(self, test_paths, test_trajectories, network, eval_mode='trajectory'):
        '''
        Evaluates the prediction model on a set of test paths.
        Origin and destination of each test path are fed to the prediction model. The predicted path is then evalauted against the ground truth path using SSPD as a metric.
        Returns a dataframe with all evaluated predictions and a summary of the model performance
        '''
        connections = network.waypoint_connections
        waypoints = network.waypoints
        
        # prepare dataframe for results
        evaluation_results = pd.DataFrame(columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD'])
        
        print(f'Evaluating model on {len(test_paths)} samples')
        print(f'Progress:', end=' ', flush=True)
        count = 0  # initialize a counter that keeps track of the progress
        percentage = 0  # percentage of progress
        # iterate through all test examples
        for i in range (0, len(test_paths)):
            true_path = test_paths['path'].iloc[i]
            mmsi = test_paths['mmsi'].iloc[i]
            orig = true_path[0]
            dest = true_path[-1]
            predicted_path = self.predict(orig, dest)
            
            if eval_mode == 'path':
                # build edge sequence from ground truth path
                ground_truth_line = geometry_utils.node_sequence_to_linestring(true_path, connections)
                # interpolate ground truth line
                ground_truth_points = geometry_utils.interpolate_line_to_gdf(ground_truth_line, connections.crs, 100)
            else:
                # get ground truth trajectory
                trajectory = test_trajectories.get_trajectory(mmsi)
                ground_truth_line, ground_truth_points = geometry_utils.clip_trajectory_between_WPs(trajectory, orig, dest, waypoints)
            
            # build linestring from predicted path
            predicted_line = geometry_utils.node_sequence_to_linestring(predicted_path, connections)
            # interpolate predicted line
            predicted_points = geometry_utils.interpolate_line_to_gdf(predicted_line, connections.crs, 100)

            # Compute SSPD
            SSPD, d12, d21 = geometry_utils.sspd(ground_truth_line, ground_truth_points, predicted_line, predicted_points)
            distances = d12.tolist() + d21.tolist()
            
            # write results to dataframe
            temp = pd.DataFrame([[mmsi, true_path, predicted_path, distances, SSPD]], columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD'])
            evaluation_results = pd.concat([evaluation_results, temp])
            
            # report progress
            count += 1
            if count/len(test_paths) > 0.1:
                count = 0
                percentage += 10
                print(f'{percentage}%...', end='', flush=True)
                
        print('Done!')
        print('\n')
        print(f'Mean SSPD: {np.mean(evaluation_results["SSPD"]):.2f}m')
        print(f'Median SSPD: {np.median(evaluation_results["SSPD"]):.2f}m')

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot 1: Box plot of SSPD
        axes[0].boxplot(evaluation_results['SSPD'])
        axes[0].set_title('Distribution of SSPD')
        axes[0].set_ylabel('SSPD (m)')

        # Plot 2: Histogram of SSPD
        axes[1].hist(evaluation_results['SSPD'], bins=np.arange(0, 2000, 50).tolist())
        axes[1].set_title('Distribution of SSPD')
        axes[1].set_xlabel('SSPD (m)')
        plt.tight_layout()
        plt.show()
        
        return evaluation_results
        
            