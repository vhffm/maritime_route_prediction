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
        while (n_walks < 10000) & (flag == False):
            n_walks = n_walks*2
            print(f'No path was found. Retrying with more random walks {n_walks}')
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

        return normalized_sorted_predictions

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

    def evaluate_next_node(self, paths, G, connections, n_start_nodes=1, n_walks=100):
        '''
        Evaluation
        '''
        classification_results = {'correct':0, 'wrong':0, 'none':0}
        SSPDs = []
        start = time.time()
        count = 0  # initialize a counter that keeps track of the progress
        percentage = 0  # percentage of progress
        print(f'Evaluating Next Node prediction on {len(paths)} paths:')
        print(f'Progress:', end=' ', flush=True)
        # iterate over all paths
        for path in paths:
            # iterate through all path nodes
            for j in range(0, len(path)-n_start_nodes):
                start_node = path[j:j+n_start_nodes]  # set start node
                try:
                    # predict next node
                    prediction = self.predict_next_nodes(start_node, G, n_steps=1, n_predictions=1, n_walks=n_walks, verbose=False)  # predict next node
                except:
                    classification_results['none'] += 1
                    continue
                else:
                    # in case of successful prediction compute accuracy and SSPD
                    predicted_node = next(iter(prediction.keys()))  # convert node format
                    predicted_node = [x for x in predicted_node][0]  # convert node format
                    true_node = path[j+n_start_nodes]
                    if true_node == predicted_node:
                        classification_results['correct'] += 1
                        SSPDs.append(0)
                    else:
                        classification_results['wrong'] += 1
                        # compute SSPD between true and predicted node sequence
                        true_sequence = [start_node[-1], true_node]
                        ground_truth_line = geometry_utils.node_sequence_to_linestring(true_sequence, connections)
                        ground_truth_points = geometry_utils.interpolate_line_to_gdf(ground_truth_line, connections.crs)
                        predicted_sequence = [start_node[-1], predicted_node]
                        predicted_line = geometry_utils.node_sequence_to_linestring(predicted_sequence, connections)
                        predicted_points = geometry_utils.interpolate_line_to_gdf(predicted_line, connections.crs, 100)
                        SSPD, d12, d21 = geometry_utils.sspd(ground_truth_line, ground_truth_points, predicted_line, predicted_points)
                        SSPDs.append(SSPD)
            # report progress
            count += 1
            if count/len(paths) > 0.1:
                count = 0
                percentage += 10
                print(f'{percentage}%...', end='', flush=True)
        print('Done!')
        
        end = time.time()
        print(f'Time elapsed: {(end-start)/60:.2f} minutes')

        # report results
        accuracy = classification_results['correct'] / (classification_results['correct']+classification_results['wrong'])
        print(f'Number of predictions made: {classification_results["correct"]+classification_results["wrong"]}')
        print(f'Accuracy: {accuracy:.3f}')
        print(f'Number of unsuccessful predictions (no start node): {classification_results["none"]}')
        SSPDs = np.array(SSPDs)
        print(f'Mean SSPD: {np.mean(SSPDs)}m ({np.mean(SSPDs[SSPDs>0])}m)')
                              

    def plot_predictions(self, predictions, start_node, trajectory, true_path, network, end_node=None, min_passages=3, location='stavanger', opacity=0.2):
        '''
        Plots predictions on an interactive map
        ====================================
        Params:
        predictions: output of self.predict_next_node function
        start_node: Single start node or start node sequence underlying the prediction
        trajectory: original trajectory that we want to make predictions for
        true_path: the actual path in the graph belonging to the original trajectory
        network: the underlying MaritimeTrafficNetwork object
        min_passages: only edges are plotted that have at least min_passages as edge feature
        center: center point of the plotted map
        opacity: opacity of the waypoints and edges of the maritime traffic network
        ====================================
        Returns:
        map: folium map object to display
        '''
        # plot network and basemap
        map = network.map_graph(pruned=True, min_passages=min_passages, 
                                location=location, opacity=opacity)

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

        if end_node is not None:
            wps = network.waypoints[network.waypoints.clusterID==end_node]
            wps.set_geometry('geometry', inplace=True)
            map = wps[columns_points].explore(m=map, name='end node', legend=False,
                                              marker_kwds={'radius':3},
                                              style_kwds={'color':'orange', 'fillColor':'orange', 'fillOpacity':1, 'opacity':1})
            wps.set_geometry('convex_hull', inplace=True, crs=network.crs)
            map = wps[columns_hull].explore(m=map, name='end_node convex hull', legend=False,
                                            style_kwds={'color':'orange', 'fillColor':'orange', 'fillOpacity':0.3, 'opacity':1})
            
        
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


        