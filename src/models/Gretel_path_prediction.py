import itertools
import logging
import os
import sys
import time
from collections import Counter
import heapq
import matplotlib.pyplot as plt
import numpy as np
import tensorboardX
import torch
from termcolor import colored
from typing import Optional, List, Callable

from gretel.config import Config, config_generator
from gretel.graph import Graph
from gretel.metrics import Evaluator
from gretel.model import Model, MLP, MultiDiffusion, EdgeTransformer
from gretel.trajectories import Trajectories
from gretel.utils import generate_masks
from gretel.main import *

import warnings
warnings.filterwarnings('ignore')

class GretelPathPrediction:

    def __init__(self):
        self.model = None
        self.graph = None
        self.config = None
        self.type = 'Gretel'
        self.task = None
        self.train_metrics = {'target_probability':[], 'choice_accuracy':[], 'choice_accuracy_deg3':[], 
                              'precision_top1':[], 'precision_top5':[], 'path_nll':[], 'path_nll_deg3':[]}
        self.test_metrics = {'target_probability':[], 'choice_accuracy':[], 'choice_accuracy_deg3':[], 
                              'precision_top1':[], 'precision_top5':[], 'path_nll':[], 'path_nll_deg3':[]}

    def train(self, config_file, directory, task):
        '''
        Trains a Gretel model based on observed paths in a network
        ====================================
        Params:
        config_file: name of the configuration file with all model parameters
        directory: path to a folder that has to contain the following files:
                    * configuration file with name config_file (parameter above)
                    * folder with name specified in config_file (input_directory) containing the following files:
                        - nodes.txt & edges.txt : contain graph data
                        - lengths.txt, observations.txt & paths.txt : contain path data
                        for more details on these files, check the notebook M4_data_format_for_gretel.ipynb
        task: the prediction task the model is trained for (e.g. 'start-to-end' or 'next-node')
        '''
        self.task = task
        # load configuration from file
        config = Config()
        config.load_from_file(directory+config_file)

        # load graph and trajectories from file
        graph, trajectories, pairwise_node_features, _ = load_data(config)
        train_trajectories, valid_trajectories, test_trajectories = trajectories
        use_validation_set = len(valid_trajectories) > 0
        graph = graph.to(config.device)
        
        given_as_target, siblings_nodes = None, None
        if pairwise_node_features is not None:
            pairwise_node_features = pairwise_node_features.to(config.device)

        print(f'==== START "{config.name}" ====')
        
        torch.manual_seed(config.seed)
        
        if config.enable_checkpointing:
            chkpt_dir = os.path.join(config.workspace, config.checkpoint_directory, config.name)
            os.makedirs(chkpt_dir, exist_ok=True)
            print(f"Checkpoints will be saved in [{chkpt_dir}]")
        
        d_node = graph.nodes.shape[1] if graph.nodes is not None else 0
        d_edge = graph.edges.shape[1] if graph.edges is not None else 0
        print(f"Number of node features {d_node}. Number of edge features {d_edge}")
        
        model = create_model(graph, pairwise_node_features, config)
        model = model.to(config.device)
        
        optimizer = create_optimizer(model.parameters(), config)
        
        if config.restore_from_checkpoint:
            filename = input("Checkpoint file: ")
            checkpoint_data = torch.load(filename)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            print("Loaded parameters from checkpoint")
        
        def create_evaluator():
            return Evaluator(
                graph.n_node,
                given_as_target=given_as_target,
                siblings_nodes=siblings_nodes,
                config=config,
            )
        
        if use_validation_set:
            valid_evaluator = Evaluator(
                graph.n_node,
                given_as_target=given_as_target,
                siblings_nodes=siblings_nodes,
                config=config,
            )
        
        if config.compute_baseline:
            display_baseline(config, graph, train_trajectories, test_trajectories, create_evaluator())
        
        graph = graph.add_self_loops(
            degree_zero_only=config.self_loop_deadend_only, edge_value=config.self_loop_weight
        )
        
        if config.rw_non_backtracking:
            print("Computing non backtracking graph...", end=" ")
            sys.stdout.flush()
            graph.compute_non_backtracking_edges()
            print("Done")
        
        evaluate(
            model, graph, test_trajectories, pairwise_node_features, create_evaluator, dataset="TEST")
        
        for epoch in range(config.number_epoch):
        
            print(f"\n=== EPOCH {epoch} ===")
        
            model.train()
            train_epoch(model, graph, optimizer, config, train_trajectories, pairwise_node_features)
        
            # TRAIN and TEST metrics computation
            '''
            train_evaluator = evaluate(
                model,
                graph,
                train_trajectories,
                pairwise_node_features,
                create_evaluator,
                dataset="TRAIN",
            )
            '''
            
            test_evaluator = evaluate(
                model,
                graph,
                test_trajectories,
                pairwise_node_features,
                create_evaluator,
                dataset="TEST",
            )

            # append train and test metrics to dictionary
            '''
            for key, v in train_evaluator.metrics.items():
                self.train_metrics[key].append(v.mean())
            '''
            for key, v in test_evaluator.metrics.items():
                self.test_metrics[key].append(v.mean())
            
            valid_evaluator = None
            if use_validation_set:
                valid_evaluator = evaluate(
                    model,
                    graph,
                    valid_trajectories,
                    pairwise_node_features,
                    create_evaluator,
                    dataset="EVAL",
                )
                
            # Logging and saving checkpoints
            if config.enable_checkpointing and epoch % config.chechpoint_every_num_epoch == 0:
                print("Checkpointing...")
                directory = os.path.join(config.workspace, config.checkpoint_directory, config.name)
                chkpt_file = os.path.join(directory, f"{epoch:04d}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    chkpt_file,
                )
                config_file = os.path.join(directory, "config")
                config.save_to_file(config_file)
        
                metrics_file = os.path.join(directory, f"{epoch:04d}.txt")
                with open(metrics_file, "w") as f:
                    f.write(test_evaluator.to_string())
                    if valid_evaluator:
                        f.write("\n\n=== VALIDATION ==\n\n")
                        f.write(valid_evaluator.to_string())
        
                print(colored(f"Checkpoint saved in {chkpt_file}", "blue"))

        # Training finished
        self.graph = graph
        self.model = model
        self.config = config

    def sample_paths(self, start_nodes, n_walks=200, max_path_length=100):
        '''
        Samples a certain number of paths from a start node
        ====================================
        Params:
        start_node: Single start node 
        n_samples: number of samples to predict
        max_path_length: maximum length of the path to be predicted (# of subsequent nodes)
        ====================================
        Returns:
        normalized_samples: Dictionary of paths and their probability of occurence, sorted by probability (descending)
        '''
        # build a tensor of shape (len(start_node), n_nodes) that is 1 for the start node and 0 otherwise
        n_nodes = self.graph.n_node
        if isinstance(start_nodes, int):
            start_nodes = [start_nodes]
        observations = torch.zeros(len(start_nodes), n_nodes)
        for i, start_node in enumerate(start_nodes):    
            observations[i][start_node] = 1
        
        # masks that contain the number of observations, start nodes and target nodes
        # modify this for more than one start node
        observed, starts, targets = generate_masks(
                    trajectory_length=observations.shape[0]+1,
                    number_observations=self.config.number_observations,
                    predict=self.config.target_prediction,
                    with_interpolation=self.config.with_interpolation,
                    device=self.config.device,
                )
        
        # compute diffusions
        diffusion_graph = self.graph
        virtual_coords = self.model.compute_diffusion(diffusion_graph, observations)
        if self.model.double_way_diffusion:
            virtual_coords_reversed = self.model.compute_diffusion(
                diffusion_graph.reverse_edges(), observations
            )
            virtual_coords = torch.cat([virtual_coords, virtual_coords_reversed])
        
        # compute rw graph
        rw_graphs = self.model.compute_rw_weights(
            virtual_coords, observed, None, targets, self.graph
        )
        rw_weights = rw_graphs.edges.transpose(0, 1)
        rw_graph = rw_graphs.update(edges=rw_weights[0])
        # MAKE PREDICTIONS
        predictions = rw_graph.sample_random_walks(start_node, 
                                                   num_samples=n_walks, 
                                                   num_steps=max_path_length, 
                                                   allow_backward=False)
        # predictions[0] contains node sequence, predictions[1] the corresponding edge sequence. We only keep the former
        sampled_paths_raw = predictions[0].tolist()
        # if a path is predicted that is shorter than max_path_length, the list is filled with -1. These need to be cut off.
        sampled_paths = [start_nodes[:-1] + [element for element in sublist if element != -1] for sublist in sampled_paths_raw]
        
        # Convert each predicted node sequence to a tuple and count occurrences
        counter_dict = Counter(map(tuple, sampled_paths))
        samples_dict = {key: value for key, value in counter_dict.items()}  # convert to dictionary
        # Sort the dictionary number of occurences
        sorted_samples = dict(sorted(samples_dict.items(), key=lambda item: item[1], reverse=True))
        # Normalize to get probabilities
        n_paths = len(predictions[0])
        normalized_samples = {key: value / n_paths for key, value in sorted_samples.items()}

        return normalized_samples

    def predict_next_nodes(self, start_nodes, n_steps=1, n_predictions=1, n_walks=200):
        '''
        Given a start node or a start node sequence, the model predicts the next node(s)
        ====================================
        Params:
        start_node: Single start node or start node sequence, for example: [1], or [1, 2, 3]
                    Sequence cannot be longer than max_order of the MOGen model
        G: the graph underlying the traffic network (networkx graph object)
        n_predictions: number of node candidates to predict
        n_steps: the maximum length of the predicted path ahead
        n_walks: the number of random walks performed by the MOGen model. The higher this number, the better the prediction of next node(s)
        ====================================
        Returns:
        predictions: dictionary of nodes sequences and their predicted probabilities
        '''
        if n_steps < 1:
            print('Number of steps n_steps needs to be > 0. Setting n_steps to 100.')
            n_steps = 100
        # predict n_walks path from start_node
        predicted_paths = self.sample_paths(start_nodes, n_walks=n_walks, max_path_length=n_steps)
        index = len(start_nodes)
        sums_dict = {}
        for key, val in predicted_paths.items():
            node_sequence = key
            if node_sequence[-1] == node_sequence[-2]:
                node_sequence = node_sequence[:-1]
            # Either append path to the dictionary of predictions...
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
    
    def predict_path(self, start_nodes, end_node, max_path_length=100, n_predictions=1, n_walks=200):
        '''
        Given a start node and an end node, the model predicts likely paths between these
        ====================================
        Params:
        start_node: node ID of single start node, e.g. 1
        end_node: node ID of end node, e.g. 5
        n_predictions: number of path candidates to predict
        n_walks: the number of random walks performed by the model. The higher this number, the better the prediction
        ====================================
        Returns:
        predictions: dictionary of paths and their predicted probabilities
        '''
        # predict n_walks paths from start_node
        predicted_paths = self.sample_paths(start_nodes, n_walks, max_path_length)
        sums_dict, flag = self.return_valid_paths(predicted_paths, start_nodes, end_node, n_walks)
        while (n_walks < 10000) & (flag == False):
            n_walks = n_walks*2
            #print(f'No path was found. Retrying with more random walks {n_walks}')
            predicted_paths = self.sample_paths(start_nodes, n_walks)
            sums_dict, flag = self.return_valid_paths(predicted_paths, start_nodes, end_node, n_walks)
        # only retain the desired number of predicted alternatives
        if n_predictions == -1:
            predictions = heapq.nlargest(len(sums_dict), sums_dict.items(), key=lambda x: x[1])
        else:
            predictions = heapq.nlargest(np.min([len(sums_dict), n_predictions]), sums_dict.items(), key=lambda x: x[1])
        
        # convert to dictionary and sort
        predictions = dict(predictions)
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        # normalize observed predictions, to get probabilities
        total_sum = sum(sums_dict.values())
        normalized_sorted_predictions = {key: value / total_sum for key, value in sorted_predictions.items()}

        return normalized_sorted_predictions, flag

    def return_valid_paths(self, predicted_paths, start_node, end_node, n_walks):
        sums_dict = {}
        flag = False
        for key, val in predicted_paths.items():
            node_sequence = key
            # check if the predicted path is valid and contains the end node
            if end_node in node_sequence:
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

    def plot_train_test_metrics(self, test_only=False):
        '''
        Plot metrics for training and test set for each epoch
        '''
        # Create subplots
        num_keys = len(self.test_metrics)
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
        
        # Loop through each key and create a subplot
        for i, key in enumerate(self.test_metrics.keys()):
            if test_only==False:
                train_values = self.train_metrics[key]
            test_values = self.test_metrics[key]
            x = np.arange(len(test_values))
            
            # get the axis to plot on
            row, col = divmod(i, 3)
            ax = axes[row, col]
            
            # plot
            if test_only==False:
                ax.plot(x, train_values, label='Train', color='b')
            ax.plot(x, test_values, label='Test', color='r')

            # set axis limits
            if key in ['target_probability', 'choice_accuracy', 'choice_accuracy_deg3', 
                       'precision_top1', 'precision_top5']:
                ax.set_ylim([0,1])
            
            # add labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel(key)
            ax.set_title(key)
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    def predict(self, start_node, end_node, number_steps):
        ######## BROKEN #########
        # build a tensor of shape (n_nodes, ) that is 1 for the start node and 0 otherwise
        number_steps = torch.tensor([number_steps])
        n_nodes = self.graph.n_node
        observations = torch.zeros(2, n_nodes)
        observations[0][start_node] = 1
        observations[1][end_node] = 1
        
        # masks that contain the number of observations, start nodes and target nodes
        observed, starts, targets = generate_masks(
                    trajectory_length=observations.shape[0],
                    number_observations=self.config.number_observations,
                    predict=self.config.target_prediction,
                    with_interpolation=self.config.with_interpolation,
                    device=self.config.device,
                )
        print(observations.shape[0])
        print(number_steps)
        print(observations)
        print(observed)
        print(starts)
        print(targets)
        
        # compute diffusions
        diffusion_graph = (
                    self.graph if not self.config.diffusion_self_loops else self.graph.add_self_loops()
                )

        # check shapes
        assert observed.shape[0] == starts.shape[0] == targets.shape[0]
        n_pred = observed.shape[0]

        # baseline
        if self.model.diffusion_graph_transformer is None and self.model.direction_edge_mlp is None:
            # if not a random graph, take the uniform random graph
            if (
                self.graph.edges.shape != torch.Size([self.graph.n_edge, 1])
                or ((self.graph.out_degree - 1.0).abs() > 1e-5).any()
            ):
                rw_graphs = self.graph.update(
                    edges=torch.ones([self.graph.n_edge, n_pred], device=self.graph.device)
                )
                rw_graphs = rw_graphs.softmax_weights()
            else:
                rw_graphs = self.graph
            virtual_coords = None
        else:
            # compute diffusions
            virtual_coords = self.model.compute_diffusion(diffusion_graph, observations)
            if self.model.double_way_diffusion:
                virtual_coords_reversed = self.model.compute_diffusion(
                    diffusion_graph.reverse_edges(), observations
                )
                virtual_coords = torch.cat([virtual_coords, virtual_coords_reversed])

            # compute rw graph
            rw_graphs = self.model.compute_rw_weights(
                virtual_coords, observed, None, targets, self.graph
            )

        n_pred = len(starts)
        n_node = observations.shape[1]
        device = observations.device
        rw_weights = rw_graphs.edges.transpose(0, 1)

        start_distributions = observations[starts]  # batch x n_node
        rw_steps = self.model.compute_number_steps(starts, targets, number_steps)

        predict_distributions = torch.zeros(n_pred, n_node, device=device)

        for pred_id in range(n_pred):
            rw_graph = rw_graphs.update(edges=rw_weights[pred_id])

            max_step_rw = None
            if self.model.rw_expected_steps:
                max_step_rw = rw_steps[pred_id]

            start_nodes = start_distributions[pred_id]
            print(start_nodes[self.graph.senders].shape)
            print(self.graph.edges.shape)
            if self.model.rw_non_backtracking:
                edge_start = start_nodes[self.graph.senders] * self.graph.edges.transpose(0,1)
                edge_signal = edge_start
                for _ in range(number_steps[0] - 1):
                    edge_signal = rw_graph @ edge_signal

                node_signal = scatter_add(
                    src=edge_signal, index=self.graph.receivers, dim_size=self.n_node)

                
            else:
                predict_distributions[pred_id] = rw_graph.random_walk(start_nodes, max_step_rw)
        
        '''
        # make prediction
        predictions, _, rw_weights = self.model(
                    observations,
                    self.graph,
                    diffusion_graph,
                    observed=observed,
                    starts=starts,
                    targets=targets,
                    pairwise_node_features=None,
                    number_steps=number_steps,
                )
        '''

        #return predictions, rw_weights
        return node_signal, edge_signal