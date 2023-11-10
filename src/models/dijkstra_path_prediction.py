import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
from datetime import timedelta, datetime
from shapely.geometry import Point, LineString, MultiLineString
from shapely import ops
import networkx as nx
import matplotlib.pyplot as plt
import folium
import time
import warnings
import pickle
import sys

# add paths for modules
sys.path.append('../visualization')
sys.path.append('../features')

# import modules
import visualize
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
                ground_truth_line = []
                for j in range(0, len(true_path)-1):
                    line = connections[(connections['from'] == true_path[j]) & (connections['to'] == true_path[j+1])].geometry.item()
                    ground_truth_line.append(line)
                ground_truth_line = MultiLineString(ground_truth_line)
                ground_truth_line = ops.linemerge(ground_truth_line)
                # interpolate ground truth line
                ground_truth_points = [ground_truth_line.interpolate(dist) for dist in range(0, int(ground_truth_line.length)+1, 50)]
                ground_truth_points_coords = [Point(point.x, point.y) for point in ground_truth_points]  # interpolated points on edge sequence
                ground_truth_points = pd.DataFrame({'geometry': ground_truth_points_coords})
                ground_truth_points = gpd.GeoDataFrame(ground_truth_points, geometry='geometry', crs=connections.crs)
            else:
                # build edge sequence from ground truth trajectory
                trajectory = test_trajectories.get_trajectory(mmsi)
                #ground_truth_line = trajectory.to_traj_gdf()
                ground_truth_points = trajectory.to_point_gdf()
                # clip trajectory to the segment between origin and destination waypoint
                WP1 = waypoints[waypoints.clusterID==orig]['geometry'].item()  # coordinates of waypoint at beginning of edge sequence
                WP2 = waypoints[waypoints.clusterID==dest]['geometry'].item()  # coordinates of waypoint at end of edge sequence
                idx1 = np.argmin(WP1.distance(ground_truth_points.geometry))  # index of trajectory point closest to beginning of edge sequence
                idx2 = np.argmin(WP2.distance(ground_truth_points.geometry))  # index of trajectory point closest to end of edge sequence
                if idx2 <= idx1:  # for roundtrips
                    print('roundtrip', idx1, idx2)
                    idx1 = 0
                    idx2 = -1
                t1 = ground_truth_points.index[idx1]
                t2 = ground_truth_points.index[idx2]
                ground_truth_line = trajectory.get_linestring_between(t1, t2)  # trajectory associated with the edge sequence
                ground_truth_points = ground_truth_points.iloc[idx1:idx2]  # trajectory points associated with the edge sequence
            
            # build edge sequence from predicted path
            predicted_line = []
            for j in range(0, len(predicted_path)-1):
                line = connections[(connections['from'] == predicted_path[j]) & (connections['to'] == predicted_path[j+1])].geometry.item()
                predicted_line.append(line)
            predicted_line = MultiLineString(predicted_line)
            predicted_line = ops.linemerge(predicted_line)
            # interpolate predicted line
            predicted_points = [predicted_line.interpolate(dist) for dist in range(0, int(predicted_line.length)+1, 50)]
            predicted_points_coords = [Point(point.x, point.y) for point in predicted_points]  # interpolated points on edge sequence
            predicted_points = pd.DataFrame({'geometry': predicted_points_coords})
            predicted_points = gpd.GeoDataFrame(predicted_points, geometry='geometry', crs=connections.crs)

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
        
            