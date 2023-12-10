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


def evaluate(model, prediction_task, test_paths, test_trajectories, network, n_start_nodes=1, MOGen_n_walks=100, MOGen_n_steps=1, eval_mode='path'):
    '''
    Evaluates the prediction model on a set of test paths.
    Origin and destination of each test path are fed to the prediction model. The predicted path is then evalauted against the ground truth path using SSPD as a metric.
    Returns a dataframe with all evaluated predictions and a summary of the model performance
    model_type: 'MOGen' or 'Dijkstra'
    prediction_type: 'next_nodes', 'path'
    '''
    supported_model_types = {'path':['Dijkstra', 'MOGen'],
                             'next_nodes':['MOGen']}
    
    connections = network.waypoint_connections.copy()
    waypoints = network.waypoints.copy()
    G = network.G.copy()

    
    # prepare dataframe for results
    evaluation_results = pd.DataFrame(columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD'])
    
    start = time.time()
    print(f'Evaluating {model.type} model on {len(test_paths)} samples for {prediction_task} prediction task')
    print(f'Progress:', end=' ', flush=True)
    count = 0  # initialize a counter that keeps track of the progress
    percentage = 0  # percentage of progress
    # iterate through all test examples
    for i in range (0, len(test_paths)):
        true_path = test_paths['path'].iloc[i]
        mmsi = test_paths['mmsi'].iloc[i]
        start_node = true_path[0:n_start_nodes]

        # MAKE PREDICTIONS
        # Prediction task: path from start to end node
        if prediction_task == 'path':
            end_node = true_path[-1]
            if model.type == 'MOGen':
                try:
                    prediction, flag = model.predict_path(start_node, end_node, G, n_predictions=1, n_walks=MOGen_n_walks, verbose=False)
                except:
                    predicted_path, SSPD, distances = [], np.nan, np.nan
                else:
                    if flag:
                        predicted_path = start_node + [x for x in list(prediction)[0]]  # appends the start node to the predicted path
                        SSPD, distances = compute_sspd(eval_mode, true_path, predicted_path, test_trajectories, mmsi, connections, start_node[0], end_node, waypoints)
                    else:
                        predicted_path, SSPD, distances = [], np.nan, np.nan
            elif model.type == 'Dijkstra':
                start = start_node[-1] # since Dijkstra can only handle one start node, we take the last node of the start node sequence as start node
                predicted_path = model.predict(start, end_node)
                predicted_path = start_node[0:-1] + predicted_path  # prepend remainder of start node
                SSPD, distances = compute_sspd(eval_mode, true_path, predicted_path, test_trajectories, mmsi, connections, start_node[0], end_node, waypoints)
            else:
                print(f'{model.type} is not a supported model type for prediction type {prediction_task}. Supported model types are {supported_model_types[prediction_type]}')
                return False
        # Prediction task: next node(s)         
        elif prediction_task == 'next_nodes':
            if model.type == 'MOGen':
                try:
                    prediction = model.predict_next_nodes(start_node, G, n_steps=MOGen_n_steps, n_predictions=1, n_walks=MOGen_n_walks, verbose=False)  # predict next node
                except:
                    predicted_path, SSPD, distances = [], np.nan, np.nan
                else:
                    predicted_path = start_node + [x for x in list(prediction)[0]]  # appends the start node to the predicted path
                    end_node = predicted_path[-1]
                    true_path = true_path[0:n_start_nodes+1]
                    if predicted_path == true_path:
                        SSPD, distances = 0, [0]
                    else:
                        SSPD, distances = compute_sspd(eval_mode, true_path, predicted_path, test_trajectories, mmsi, connections, start_node[0], end_node, waypoints)
            else:
                print(f'{model.type} is not a supported model type for prediction type {prediction_task}. Supported model types are {supported_model_types[prediction_type]}')
                return False
        
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
    end = time.time()
    print(f'Time elapsed: {(end-start)/60:.2f} minutes')
    print('\n')

    # find number of unsuccessful predictions
    nan_mask = evaluation_results.isna().any(axis=1)
    
    print(f'Percentage of unsuccessful predictions: {nan_mask.sum() / len(evaluation_results)*100:.2f}%')
    print(f'Mean SSPD: {np.mean(evaluation_results[~nan_mask]["SSPD"]):.2f}m')
    print(f'Median SSPD: {np.median(evaluation_results[~nan_mask]["SSPD"]):.2f}m')

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

def evaluate_given_predictions(prediction_task, path_df, test_trajectories, network, n_start_nodes=1, eval_mode='path'):
    '''
    Evaluates a set of true paths against predicted paths
    Origin and destination of each test path are fed to the prediction model. The predicted path is then evalauted against the ground truth path using SSPD as a metric.
    Returns a dataframe with all evaluated predictions and a summary of the model performance
    prediction_type: 'next_nodes', 'path'
    '''    
    connections = network.waypoint_connections.copy()
    waypoints = network.waypoints.copy()
    G = network.G.copy()

    # prepare dataframe for results
    evaluation_results = pd.DataFrame(columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD'])
    
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

        # Prediction task: path from start to end node
        if prediction_task == 'path':
            end_node = true_path[-1]
            SSPD, distances = compute_sspd(eval_mode, true_path[n_start_nodes-1:], predicted_path[n_start_nodes-1:], 
                                           test_trajectories, mmsi, connections, start_node[-1], end_node, waypoints)
                     
        # Prediction task: next node(s)         
        elif prediction_task == 'next_nodes':
            end_node = predicted_path[-1]
            true_path = true_path[0:n_start_nodes+1]
            if predicted_path == true_path:
                SSPD, distances = 0, [0]
            else:
                SSPD, distances = compute_sspd(eval_mode, true_path[n_start_nodes-1:], predicted_path[n_start_nodes-1:], 
                                               test_trajectories, mmsi, connections, start_node[-1], end_node, waypoints)
        
        # write results to dataframe
        temp = pd.DataFrame([[mmsi, true_path, predicted_path, distances, SSPD]], columns=['mmsi', 'true_path', 'predicted_path', 'distances', 'SSPD'])
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
    '''
    # EVALUATE PREDICTIONS
    if eval_mode == 'path':
        # build edge sequence from ground truth path
        ground_truth_line = geometry_utils.node_sequence_to_linestring(true_path, connections)
        # interpolate ground truth line
        ground_truth_points = geometry_utils.interpolate_line_to_gdf(ground_truth_line, connections.crs, 100)
    else:
        # get ground truth trajectory
        trajectory = test_trajectories.get_trajectory(mmsi)
        ground_truth_line, ground_truth_points = geometry_utils.clip_trajectory_between_WPs(trajectory, start_node, end_node, waypoints)
    
    # build linestring from predicted path
    predicted_line = geometry_utils.node_sequence_to_linestring(predicted_path, connections)
    # interpolate predicted line
    predicted_points = geometry_utils.interpolate_line_to_gdf(predicted_line, connections.crs, 100)

    # Compute SSPD
    SSPD, d12, d21 = geometry_utils.sspd(ground_truth_line, ground_truth_points, predicted_line, predicted_points)
    distances = d12.tolist() + d21.tolist()
    return SSPD, distances