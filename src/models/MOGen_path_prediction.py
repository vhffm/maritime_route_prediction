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

    def train(self, paths, max_order, model_selection=True):
        '''
        Trains a MOGen model based on observed paths in a network
        ====================================
        Params:
        paths: List of paths, for example: [[2, 3, 4], [1, 7, 9]]
        max_order: The maximum order of the MOGen model. 
                   If we want to make predictions specifying a sequence of n start nodes, max_order needs to be at least n.
        model_selection: If True, then MOGen trains a model up to max_order, including all models < max_order
                         If False, only a model with max_order is trained
        ====================================    
        '''
        # create a pathpy path collection and add all training paths to it
        pc = pp.PathCollection()
        for raw_path in paths:
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

    def predict_paths(self, start_node, n_paths, seed=42):
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
        predictions = self.model.predict(no_of_paths=n_paths, max_order=self.model.max_order, start_node=start_node_str, seed=seed)
        # sort predictions by frequency of occurrence
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        sorted_predictions_dict={}
        for key, val in sorted_predictions.items():
            node_sequence = tuple(int(x) for x in key)
            sorted_predictions_dict[node_sequence] = val/n_paths  # probability of occurrence
        
        return sorted_predictions_dict

    def predict_next_node(self, start_node, n_steps=1, n_predictions=-1, n_walks=100):
        '''
        Given a start node or a start node sequence, the model predicts the next node(s)
        ====================================
        Params:
        start_node: Single start node or start node sequence, for example: [1], or [1, 2, 3]
                    Sequence cannot be longer than max_order of the MOGen model
        n_predictions: number of node candidates to predict
        n_steps: the maximum length of the predicted path ahead (if -1, then we look at path predictions of unlimited length)
        n_walks: the number of random walks performed by the MOGen model. The higher this number, the better the prediction of next node(s)
        ====================================
        Returns:
        predictions: dictionary of nodes and their predicted probabilities
        '''
        # predict n_paths from start_node
        predicted_paths = self.predict_paths(start_node, n_walks)
        index = len(start_node)
        sums_dict = {}
        for key, val in predicted_paths.items():
            if n_steps > 0:
                prefix = key[index:index+n_steps]
            else:
                prefix = key[index:]
            if prefix not in sums_dict:
                sums_dict[prefix] = val
            else:
                sums_dict[prefix] += val
        if n_predictions == -1:
            predictions = heapq.nlargest(len(sums_dict), sums_dict.items(), key=lambda x: x[1])
        else:
            predictions = heapq.nlargest(np.min([len(sums_dict), n_predictions]), sums_dict.items(), key=lambda x: x[1])
        predictions = dict(predictions)
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
         
        return sorted_predictions

    def plot_predictions(self, predictions, start_node, trajectory, true_path, network, min_passages=3, center=[59, 5], opacity=0.2):
        # plot network and basemap
        map = network.map_graph(pruned=True, min_passages=min_passages, 
                                center=center, opacity=opacity)

        # highlight the start node(s)
        wps = network.waypoints[network.waypoints.clusterID.isin(start_node)]
        columns_points = ['clusterID', 'geometry', 'speed', 'cog_before', 'cog_after', 'n_members']  # columns to plot
        columns_hull = ['clusterID', 'convex_hull', 'speed', 'cog_before', 'cog_after', 'n_members']  # columns to plot
        wps.set_geometry('geometry', inplace=True)
        map = wps[columns_points].explore(m=map, name='start nodes', legend=False,
                                          marker_kwds={'radius':3},
                                          style_kwds={'color':'yellow', 'fillColor':'yellow', 'fillOpacity':1, 'opacity':1})
        wps.set_geometry('convex_hull', inplace=True, crs=network.crs)
        map = wps[columns_hull].explore(m=map, name='start nodes convex hulls', legend=False,
                                        style_kwds={'color':'yellow', 'fillColor':'yellow', 'fillOpacity':0.3, 'opacity':1})
        
        # generate plottable dataframe of predicted paths
        predicted_paths = pd.DataFrame(columns=['path', 'geometry', 'probability'])
        for key, value in predictions.items():
            path = start_node + [x for x in key]
            line = geometry_utils.node_sequence_to_linestring(path, network.waypoint_connections)
            temp = pd.DataFrame([[tuple(path), line, value]], columns=['path', 'geometry', 'probability'])
            predicted_paths = pd.concat([predicted_paths, temp])
        predicted_paths = gpd.GeoDataFrame(predicted_paths, geometry='geometry', crs=network.crs)
        
        # plot prediction and ground truth
        true_path_line = geometry_utils.get_geo_df(true_path, network.waypoint_connections)
        trajectory = trajectory.to_line_gdf()
        mmsi = trajectory['mmsi'].unique()

        map = trajectory[['mmsi', 'geometry', 'skipsgruppe', 'length', 'bredde']].explore(m=map, style_kwds={'weight':3, 'color':'black', 'opacity':1},
                                                                                          name=f'{mmsi} trajectory')
        map = true_path_line.explore(m=map, style_kwds={'weight':3, 'color':'cyan', 'opacity':1},
                                   name=f'{mmsi} closest path')
        for i in range (0, len(predicted_paths)):
            map = predicted_paths.iloc[[i,i]].explore(m=map, style_kwds={'weight':3, 'color':'yellow', 'opacity':1},
                                                  name=f'Prediction {i} ({predicted_paths["probability"].iloc[i]*100}% probability)')
        folium.LayerControl().add_to(map)
        return map


        