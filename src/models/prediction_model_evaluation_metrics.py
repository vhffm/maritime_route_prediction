'''
utility functions for prediction model evaluation
'''

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
import pathpy as pp
import heapq
import pickle

# add paths for modules
sys.path.append('../features')
import geometry_utils


def evaluate_given_predictions(prediction_task, path_df, test_trajectories, network, n_start_nodes=1, n_steps=1, eval_mode='path'):
    '''
    Evaluates a set of predicted paths against ground truth paths or trajectories.
    The computed metrics are choice accuracy CACC, mean SSPD and median SSPD
    ====================================
    Params:
    prediction_task: 'path' for subtask 1 and 'next_nodes' for subtask 2
    path_df: (GeoDataframe) contains predicted and ground truth paths
    test_trajectories: (MovingPandas Trajectory collection) the ground truth trajectories
    network: (MaritimeTrafficNetwork object) the underlying MTN
    n_start_nodes: (int) number of observed nodes in the path prefix
    n_steps: (int) prediction horizon (only for subtask 2)
    eval_mode: (str) 'path' for evaluation against ground truth paths
                     'trajectory' for evaluation against ground truth trajectories
    ====================================
    Returns:
    evaluation_results: (Dataframe) evaluation metrics  
    fig: (matplotlib fuigure object) plots of evaluation metrics
    '''    
    # extract waypoints, connections and graph from MTN
    connections = network.waypoint_connections.copy()
    waypoints = network.waypoints.copy()
    G = network.G.copy()

    # prepare dataframe for results
    evaluation_results = pd.DataFrame(columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD', 'choice_accuracy'])
    
    start = time.time()
    print(f'Evaluating {len(path_df)} samples for {prediction_task} prediction task')
    print(f'Progress:', end=' ', flush=True)
    count = 0  # initialize a counter that keeps track of the progress
    percentage = 0  # percentage of progress
    # iterate through all test examples
    for i in range (0, len(path_df)):
        true_path = path_df['ground_truth'].iloc[i]
        predicted_path = path_df['prediction'].iloc[i]
        mmsi = path_df['mmsi'].iloc[i]
        start_node = true_path[0:n_start_nodes]

        if predicted_path == []:
            SSPD, distances, choice_accuracy = np.nan, [], np.nan
            temp = pd.DataFrame([[mmsi, true_path, predicted_path, distances, SSPD, choice_accuracy]], 
                            columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD', 'choice_accuracy'])
            evaluation_results = pd.concat([evaluation_results, temp])
            continue

        # Prediction task: path from start to end node (with destination information)
        if prediction_task == 'path':
            end_node = true_path[-1]
            try:
                SSPD, distances = compute_sspd(eval_mode, true_path[n_start_nodes-1:], predicted_path[n_start_nodes-1:], 
                                               test_trajectories, mmsi, connections, start_node[-1], end_node, waypoints)
                choice_accuracy = compute_choice_accuracy(true_path[n_start_nodes-1:], predicted_path[n_start_nodes-1:])
            except:
                SSPD, distances = compute_sspd(eval_mode, true_path, predicted_path, 
                                               test_trajectories, mmsi, connections, start_node[-1], end_node, waypoints)
                choice_accuracy = compute_choice_accuracy(true_path, predicted_path)
                     
        # Prediction task: next node(s)         
        elif prediction_task == 'next_nodes':
            end_node = predicted_path[-1]
            true_path = true_path[0:n_start_nodes+n_steps]
            if predicted_path == true_path:
                SSPD, distances = 0, [0]
                choice_accuracy = 1.0
            else:
                SSPD, distances = compute_sspd(eval_mode, true_path[n_start_nodes-1:], predicted_path[n_start_nodes-1:], 
                                               test_trajectories, mmsi, connections, start_node[-1], end_node, waypoints)
                choice_accuracy = compute_choice_accuracy(true_path[n_start_nodes-1:], predicted_path[n_start_nodes-1:])
        
        # write results to dataframe
        temp = pd.DataFrame([[mmsi, true_path, predicted_path, distances, SSPD, choice_accuracy]], 
                            columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD', 'choice_accuracy'])
        evaluation_results = pd.concat([evaluation_results, temp])
        
        # report progress
        count += 1
        if count/len(path_df) > 0.1:
            count = 0
            percentage += 10
            print(f'{percentage}%...', end='', flush=True)
            
    print('Done!')
    print('\n')
    end = time.time()
    print(f'Time elapsed: {(end-start)/60:.2f} minutes')
    print('\n')

    # find number of unsuccessful predictions
    nan_mask = evaluation_results.isna().any(axis=1)
    
    print(f'Percentage of unsuccessful predictions: {nan_mask.sum() / len(evaluation_results)*100:.2f}%')
    print(f'Mean SSPD: {np.mean(evaluation_results[~nan_mask]["SSPD"]):.2f}m')
    print(f'Median SSPD: {np.median(evaluation_results[~nan_mask]["SSPD"]):.2f}m')
    print(f'Mean choice_accuracy: {np.mean(evaluation_results[~nan_mask]["choice_accuracy"]):.4f}')

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Box plot of SSPD
    axes[0].boxplot(evaluation_results[~nan_mask]['SSPD'])
    axes[0].set_title('Distribution of SSPD')
    axes[0].set_ylabel('SSPD (m)')

    # Plot 2: Histogram of SSPD
    axes[1].hist(evaluation_results[~nan_mask]['SSPD'], bins=np.arange(0, 2000, 50).tolist())
    axes[1].set_title('Distribution of SSPD')
    axes[1].set_xlabel('SSPD (m)')
    plt.tight_layout()
    plt.show()
    
    return evaluation_results, fig


def compute_sspd(eval_mode, true_path, predicted_path, test_trajectories, mmsi, connections, start_node, end_node, waypoints):
    '''
    Computes the SSPD and the distribution of distance between a predicted path and a ground truth path or ground truth trajectory
    ====================================
    Params:
    eval_mode: (str) 'path' for evaluation against ground truth paths
                     'trajectory' for evaluation against ground truth trajectories
    true_path: (list of int) the ground truth path as a list of node IDs
    predicted_path: (list of int) the predicted path as a list of node IDs
    test_trajectories: (MovingPandas Trajectory collection) the ground truth trajectories
    mmsi: (str) the MMSI associated with the predicted vessel trajectory
    connections: (GeoDataframe) the edges of the MTN represented as geometric objects
    start_node: (int) the last node of the observed path prefix
    end_node: (int) the last node of the predicted path
    waypoints: (GeoDataframe) the nodes of the MTN represented as geometric objects
    ====================================
    Returns:
    SSPD: (float) the SSPD between prediction and ground truth in meters
    distances: (list of float) the individual distances that constitute the SSPD
    '''
    # evaluate predictions
    if eval_mode == 'path':
        # build edge sequence from ground truth path
        ground_truth_line = geometry_utils.node_sequence_to_linestring(true_path, connections)
        # interpolate ground truth line
        ground_truth_points = geometry_utils.interpolate_line_to_gdf(ground_truth_line, connections.crs, 100)
    elif eval_mode == 'trajectory':
        # get ground truth trajectory
        trajectory = test_trajectories.get_trajectory(mmsi)
        ground_truth_line, ground_truth_points = geometry_utils.clip_trajectory_between_WPs(trajectory, start_node, end_node, waypoints)
    else:
        print('Specify a valid evaluation mode: path or trajectory')
        return
    
    # build linestring from predicted path
    predicted_line = geometry_utils.node_sequence_to_linestring(predicted_path, connections)
    # interpolate predicted line
    predicted_points = geometry_utils.interpolate_line_to_gdf(predicted_line, connections.crs, 100)

    # Compute SSPD
    SSPD, d12, d21 = geometry_utils.sspd(ground_truth_line, ground_truth_points, predicted_line, predicted_points)
    distances = d12.tolist() + d21.tolist()
    return SSPD, distances


def compute_choice_accuracy(true_path, predicted_path):
    '''
    Computes the choice accuracy between a predicted path and a ground truth path
    Choice accuracy measures how the fraction of correct decisions taken at each node of the path
    Example:
    true_path = [1, 2, 3, 4]
    predicted_path = [1, 2, 5, 4]
    choice_accuracy = 1/3  (correct decisions: 1->2, the rest are incorrect decisions)
    ====================================
    Params:
    true_path: (list of int) the ground truth path as a list of node IDs
    predicted_path: (list of int) the predicted path as a list of node IDs
    ====================================
    Returns:
    CACC: (float) the choice accuracy
    '''
    truth_pairs = [(true_path[j], true_path[j+1]) for j in range(0, len(true_path)-1)]
    prediction_pairs = [(predicted_path[j], predicted_path[j+1]) for j in range(0, len(predicted_path)-1)]

    return sum(correct_choice in prediction_pairs for correct_choice in truth_pairs) / len(truth_pairs)




