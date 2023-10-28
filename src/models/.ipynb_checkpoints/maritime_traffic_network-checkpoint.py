import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from scipy.sparse import coo_matrix
from shapely.geometry import Point, LineString, MultiLineString
from shapely import ops
from collections import OrderedDict
import networkx as nx
import time
import folium
import warnings
import sys

# add paths for modules
sys.path.append('../visualization')
sys.path.append('../features')

# import modules
import visualize
import geometry_utils

class MaritimeTrafficNetwork:
    '''
    DOCSTRING
    '''
    def __init__(self, gdf, crs):
        self.gdf = gdf
        self.crs = crs
        self.hyperparameters = {}
        self.trajectories = mpd.TrajectoryCollection(self.gdf, traj_id_col='mmsi', obj_id_col='mmsi')
        self.significant_points_trajectory = []
        self.significant_points = []
        self.waypoints = []
        self.waypoint_connections = []
        self.waypoint_connections_pruned = []
        self.G = []
        self.G_pruned = []

    def get_trajectories_info(self):
        print(f'Number of AIS messages: {len(self.gdf)}')
        print(f'Number of trajectories: {len(self.trajectories)}')
        print(f'Coordinate Reference System (CRS): {self.gdf.crs}')

    def set_hyperparameters(self, params):
        self.hyperparameters = params
    
    def init_precomputed_significant_points(self, gdf):
        '''
        Load precomputed significant points
        '''
        print('Loading significant turning points from file...')
        self.significant_points = gdf
        self.significant_points_trajectory = mpd.TrajectoryCollection(gdf, traj_id_col='mmsi', obj_id_col='mmsi', t='date_time_utc')
        n_points, n_DP_points = len(self.gdf), len(self.significant_points)
        print(f'Number of significant points detected: {n_DP_points} ({n_DP_points/n_points*100:.2f}% of AIS messages)')

    def init_precomputed_waypoints(self, gdf):
        '''
        Load precomputed waypoints
        '''
        print('Loading precomputed waypoints from file...')
        self.waypoints = gdf
        print(f'{len(gdf)} waypoints loaded')
    
    def calc_significant_points_DP(self, tolerance):
        '''
        Detect significant turning points with Douglas Peucker algorithm 
        and add COG before and after each significant point
        :param tolerance: Douglas Peucker algorithm tolerance
        result: self.significant_points_trajectory is set to a MovingPandas TrajectoryCollection containing
                the significant turning points
                self.significant_points is set to GeoPandasDataframe containing the significant turning 
                points and COG information
        '''
        #
        print(f'Calculating significant turning points with Douglas Peucker algorithm (tolerance = {tolerance}) ...')
        start = time.time()  # start timer
        self.significant_points_trajectory = mpd.DouglasPeuckerGeneralizer(self.trajectories).generalize(tolerance=tolerance)
        n_points, n_DP_points = len(self.gdf), len(self.significant_points_trajectory.to_point_gdf())
        end = time.time()  # end timer
        print(f'Number of significant points detected: {n_DP_points} ({n_DP_points/n_points*100:.2f}% of AIS messages)')
        print(f'Time elapsed: {(end-start)/60:.2f} minutes')

        print(f'Adding course over ground before and after each turn ...')
        start = time.time()  # start timer
        self.significant_points_trajectory.add_direction()
        traj_df = self.significant_points_trajectory.to_point_gdf()
        traj_df.rename(columns={'direction': 'cog_before'}, inplace=True)
        traj_df['cog_after'] = np.nan
        for mmsi in traj_df.mmsi.unique():
            subset = traj_df[traj_df.mmsi == mmsi]
            if len(subset)>1:
                fill_value = subset['cog_before'].iloc[-1]
                subset['cog_after'] = subset['cog_before'].shift(-1, fill_value=fill_value)
            else:
                subset['cog_after'] = subset['cog_before']
            traj_df.loc[traj_df.mmsi == mmsi, 'cog_after'] = subset['cog_after'].values 
        self.significant_points = traj_df
        end = time.time()  # end timer
        print(f'Done. Time elapsed: {(end-start)/60:.2f} minutes')

    def calc_waypoints_clustering(self, method='HDBSCAN', min_samples=15, min_cluster_size=15, eps=0.008, 
                                  metric='euclidean', V=np.diag([1, 1, np.pi/18, np.pi/18])):
        '''
        Compute waypoints by clustering significant turning points
        :param method: Clustering method (supported: DBSCAN and HDBSCAN)
        :param min_points: required parameter for DBSCAN and HDBSCAN
        :param eps: required parameter for DBSCAN
        '''
        start = time.time()  # start timer
        # prepare data for clustering
        significant_points = self.significant_points
        significant_points['x'] = significant_points.geometry.x
        significant_points['y'] = significant_points.geometry.y

        # prepare clustering depending on metric
        if metric == 'euclidean':
            columns = ['x', 'y']
            metric_params = {}
        elif metric == 'haversine':
            columns = ['x', 'y']
            metric_params = {}
        elif metric == 'mahalanobis':
            columns = ['x', 'y', 'cog_before', 'cog_after', 'speed']
            metric_params = {'V':V}   # mahalanobis distance parameter matrix
            metric_params_OPTICS = {'VI':np.linalg.inv(V)}
        else:
            print(f'{metric} is not a supported distance metric. Exiting waypoint calculation...')
            return
        
        ########
        # DBSCAN
        ########
        if method == 'DBSCAN':
            print(f'Calculating waypoints with {method} (eps = {eps}, min_samples = {min_samples}) ...')
            print(f'Distance metric: {metric}')
            clustering = DBSCAN(eps=eps, min_samples=min_samples, 
                                metric=metric, metric_params=metric_params).fit(significant_points[columns])   
        #########
        # HDBSCAN
        #########
        elif method == 'HDBSCAN':
            print(f'Calculating waypoints with {method} (min_samples = {min_samples}) ...')
            print(f'Distance metric: {metric}')
            clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, 
                                 cluster_selection_method='leaf', metric=metric, 
                                 metric_params=metric_params).fit(significant_points[columns])
        #########
        # OPTICS
        #########
        elif method == 'OPTICS':
            print(f'Calculating waypoints with {method} (min_samples = {min_samples}) ...')
            print(f'Distance metric: {metric}')
            clustering = OPTICS(min_samples=min_samples, metric=metric, 
                                metric_params=metric_params_OPTICS).fit(significant_points[columns])
        else:
            print(f'{method} is not a supported clustering method. Exiting waypoint calculation...')
            return
        
        # compute cluster centroids
        cluster_centroids = pd.DataFrame(columns=['clusterID', 'lat', 'lon', 'x', 'y', 
                                                  'speed', 'cog_before', 'cog_after', 'n_members', 'convex_hull'])
        for i in range(0, max(clustering.labels_)+1):
            lat = significant_points[clustering.labels_ == i].lat.mean()
            lon = significant_points[clustering.labels_ == i].lon.mean()
            x = significant_points[clustering.labels_ == i].x.mean()
            y = significant_points[clustering.labels_ == i].y.mean()
            speed = significant_points[clustering.labels_ == i].speed.mean()
            cog_before = significant_points[clustering.labels_ == i].cog_before.mean()
            cog_after = significant_points[clustering.labels_ == i].cog_after.mean()     
            n_members = len(significant_points[clustering.labels_ == i])
            centroid = pd.DataFrame([[i, lat, lon, x, y, speed, cog_before, cog_after, n_members]], 
                                    columns=['clusterID', 'lat', 'lon', 'x', 'y', 
                                             'speed', 'cog_before', 'cog_after', 'n_members'])
            cluster_centroids = pd.concat([cluster_centroids, centroid])
        
        significant_points['clusterID'] = clustering.labels_  # assign clusterID to each waypoint
        
        # convert waypoint and cluster centroid DataFrames to GeoDataFrames
        significant_points = gpd.GeoDataFrame(significant_points, 
                                              geometry=gpd.points_from_xy(significant_points.x,
                                                                          significant_points.y), 
                                              crs=self.crs)
        significant_points.reset_index(inplace=True)
        cluster_centroids = gpd.GeoDataFrame(cluster_centroids, 
                                             geometry=gpd.points_from_xy(cluster_centroids.x,
                                                                         cluster_centroids.y),
                                             crs=self.crs)
        
        # compute convex hull of each cluster
        for i in range(0, max(clustering.labels_)+1):
            hull = significant_points[significant_points.clusterID == i].unary_union.convex_hull
            cluster_centroids['convex_hull'].iloc[i] = hull
        print(f'{len(cluster_centroids)} clusters detected')
        end = time.time()  # end timer
        print(f'Time elapsed: {(end-start)/60:.2f} minutes')
        
        # assign results
        self.significant_points = significant_points
        self.waypoints = cluster_centroids

    def merge_stop_points(self, max_speed=2):
        '''
        merges stop points (criterion max_speed) together that intersect with their convex hull
        '''
        waypoints = self.waypoints.copy()
        stop_points = waypoints[waypoints['speed'] < max_speed]
        columns = stop_points.columns.tolist()
        columns.append('members')
        merged_stop_points = pd.DataFrame(columns=columns)
        drop_IDs = []
        for i in range(0, len(stop_points)):
            current_stop_point = stop_points.iloc[i]
            mask = current_stop_point['convex_hull'].intersects(stop_points['convex_hull'])
            intersect_stop_points = stop_points[mask]
            members = intersect_stop_points['clusterID'].unique()
            drop_IDs.append(members.tolist())
            convex_hull = ops.unary_union(intersect_stop_points['convex_hull'])
            clusterID = members[0]
            lat = current_stop_point['lat']
            lon = current_stop_point['lon']
            x = current_stop_point['x']
            y = current_stop_point['y']
            speed = intersect_stop_points['speed'].mean()
            n_members = intersect_stop_points['n_members'].sum()
            geometry = Point(x, y)
            if clusterID not in merged_stop_points['clusterID'].unique():
                line = pd.DataFrame([[clusterID, lat, lon, x, y, speed, 0, 0, n_members, convex_hull, geometry, members]], 
                                    columns=columns)
                merged_stop_points = pd.concat([merged_stop_points, line])
        merged_stop_points = gpd.GeoDataFrame(merged_stop_points, geometry='geometry', crs=self.crs)
        
        # adjust waypoints
        drop_IDs = set(item for sublist in drop_IDs for item in sublist)
        drop_IDs = list(drop_IDs)
        mask = ~waypoints['clusterID'].isin(drop_IDs)
        new_waypoints = waypoints[mask]
        new_waypoints = pd.concat([new_waypoints, merged_stop_points[new_waypoints.columns]])
        new_waypoints = gpd.GeoDataFrame(new_waypoints, geometry='geometry', crs=self.crs)
        
        # adjust waypoint connections
        waypoint_connections = self.waypoint_connections.copy()
        drop_IDs = []
        for i in range(0, len(merged_stop_points)):
            merged_stop_point = merged_stop_points.iloc[i]
            merge_nodes = merged_stop_point['members']
            #print('==================')
            #print(waypoint_connections[waypoint_connections['from']==merge_nodes[0]][['from', 'to', 'passages']])
            #print(waypoint_connections[waypoint_connections['to']==merge_nodes[0]][['from', 'to', 'passages']])
            #print('Merge nodes:', merge_nodes)
            for j in range(1, len(merge_nodes)):
                #print('Current merge node: ', merge_nodes[j])
                # outgoing connections
                from_connections = waypoint_connections[waypoint_connections['from']==merge_nodes[j]]
                #print(from_connections[['from', 'to', 'passages']])
                for k in range(0, len(from_connections)):
                    #print('Current "to" node: ', from_connections['to'].iloc[k])
                    add_passages_from = from_connections['passages'].iloc[k]
                    # no self loops
                    if merge_nodes[0] == from_connections['to'].iloc[k]:
                        continue
                    mask = ((waypoint_connections['from']==merge_nodes[0]) &
                           (waypoint_connections['to']==from_connections['to'].iloc[k]))
                    if len(waypoint_connections[mask]) > 0:
                        #print(f'Increasing edge weight from {merge_nodes[0]} to {from_connections["to"].iloc[k]}')
                        waypoint_connections[mask]['passages'] += add_passages_from
                    else:
                        # add linestring as edge
                        #print(f'Adding new edge from {merge_nodes[0]} to {from_connections["to"].iloc[k]}')
                        orig = merge_nodes[0]
                        dest = from_connections['to'].iloc[k]
                        p1 = new_waypoints[new_waypoints.clusterID == orig]['geometry']
                        p2 = waypoints[waypoints.clusterID == dest]['geometry']
                        #print(p1, p2)
                        edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
                        length = edge.length
                        # compute the orientation fo the edge (COG)
                        p1 = Point(new_waypoints[new_waypoints.clusterID == orig].lon, new_waypoints[new_waypoints.clusterID == orig].lat)
                        p2 = Point(waypoints[waypoints.clusterID == dest].lon, waypoints[waypoints.clusterID == dest].lat)
                        direction = geometry_utils.calculate_initial_compass_bearing(p1, p2)
                        line = pd.DataFrame([[orig, dest, edge, direction, length, add_passages_from]], 
                                            columns=['from', 'to', 'geometry', 'direction', 'length', 'passages'])
                        waypoint_connections = pd.concat([waypoint_connections, line])
                
                #incoming connections
                to_connections = waypoint_connections[waypoint_connections['to']==merge_nodes[j]]
                #print(to_connections[['from', 'to', 'passages']])
                for k in range(0, len(to_connections)):
                    #print('Current "from" node: ', to_connections['from'].iloc[k])
                    add_passages_to = to_connections['passages'].iloc[k]
                    # no self loops
                    if merge_nodes[0] == to_connections['from'].iloc[k]:
                        continue
                    mask = ((waypoint_connections['to']==merge_nodes[0]) &
                           (waypoint_connections['from']==to_connections['from'].iloc[k]))
                    if  len(waypoint_connections[mask]) > 0:
                        #print(f'Increasing edge weight from {to_connections["to"].iloc[k]} to {merge_nodes[0]}')
                        waypoint_connections[mask]['passages'] += add_passages_to
                    else:
                        # add linestring as edge
                        #print(f'Adding new edge from {to_connections["from"].iloc[k]} to {merge_nodes[0]}')
                        dest = merge_nodes[0]
                        orig = to_connections['from'].iloc[k]
                        p1 = waypoints[waypoints.clusterID == orig]['geometry']
                        p2 = new_waypoints[new_waypoints.clusterID == dest]['geometry']
                        #print(p1, p2)
                        edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
                        length = edge.length
                        # compute the orientation fo the edge (COG)
                        p1 = Point(waypoints[waypoints.clusterID == orig].lon, waypoints[waypoints.clusterID == orig].lat)
                        p2 = Point(new_waypoints[new_waypoints.clusterID == dest].lon, new_waypoints[new_waypoints.clusterID == dest].lat)
                        direction = geometry_utils.calculate_initial_compass_bearing(p1, p2)
                        line = pd.DataFrame([[orig, dest, edge, direction, length, add_passages_to]], 
                                            columns=['from', 'to', 'geometry', 'direction', 'length', 'passages'])
                        waypoint_connections = pd.concat([waypoint_connections, line])
                drop_IDs.append(merge_nodes[j])
                #print(waypoint_connections[waypoint_connections['from']==merge_nodes[0]][['from', 'to', 'passages']])
                #print(waypoint_connections[waypoint_connections['to']==merge_nodes[0]][['from', 'to', 'passages']])
        mask = ~(waypoint_connections['to'].isin(drop_IDs) | waypoint_connections['from'].isin(drop_IDs))
        new_waypoint_connections = waypoint_connections[mask]
        #print(new_waypoint_connections[new_waypoint_connections['from']==9][['from', 'to', 'passages']])
        #print(new_waypoint_connections[new_waypoint_connections['to']==9][['from', 'to', 'passages']])
        new_waypoint_connections = gpd.GeoDataFrame(new_waypoint_connections, geometry='geometry', crs=self.crs)

        # make graph from waypoints and waypoint connections
        # add node features
        G = nx.DiGraph()
        for i in range(0, len(new_waypoints)):
            node_id = new_waypoints['clusterID'].iloc[i]
            G.add_node(node_id)
            G.nodes[node_id]['n_members'] = new_waypoints.n_members.iloc[i]
            G.nodes[node_id]['position'] = (new_waypoints.lon.iloc[i], new_waypoints.lat.iloc[i])  # !changed lat-lon to lon-lat for plotting
            G.nodes[node_id]['speed'] = new_waypoints.speed.iloc[i]
            G.nodes[node_id]['cog_before'] = new_waypoints.cog_before.iloc[i]
            G.nodes[node_id]['cog_after'] = new_waypoints.cog_after.iloc[i]
        
        for i in range(0, len(new_waypoint_connections)):
            orig = new_waypoint_connections['from'].iloc[i]
            dest = new_waypoint_connections['to'].iloc[i]
            e = (orig, dest)
            G.add_edge(*e)
            G[orig][dest]['weight'] = new_waypoint_connections['passages'].iloc[i]
            G[orig][dest]['length'] = new_waypoint_connections['length'].iloc[i]
            G[orig][dest]['direction'] = new_waypoint_connections['direction'].iloc[i]
            G[orig][dest]['geometry'] = new_waypoint_connections['geometry'].iloc[i]
            G[orig][dest]['inverse_weight'] = 1/new_waypoint_connections['passages'].iloc[i]

        self.waypoints = new_waypoints
        self.waypoint_connections = new_waypoint_connections
        self.G = G
    
    def prune_graph(self, min_passages):
        '''
        pruned the traffic graph. If an edge between nodes u and v has a weight<min_passages, 
        the edge is removed when there is an alternative path between u and v
        '''
        G_pruned = self.G.copy()
        G_temp = self.G.copy()
        connections_pruned = self.waypoint_connections.copy()

        edges_to_remove = []
        for u, v in G_pruned.edges():
            if G_pruned[u][v]['weight'] < min_passages:
                G_temp.remove_edge(u, v)
                if nx.has_path(G_temp, u, v):
                    edges_to_remove.append((u, v))
    
        # Actually remove the edges after iterating through all of them
        for u, v in edges_to_remove:
            G_pruned.remove_edge(u, v)
            connections_pruned = connections_pruned[~((connections_pruned['from'] == u) & (connections_pruned['to'] == v))]

        print('------------------------')
        print(f'Pruned Graph:')
        print(f'Number of nodes: {G_pruned.number_of_nodes()} ({nx.number_of_isolates(G_pruned)} isolated)')
        print(f'Number of edges: {G_pruned.number_of_edges()}')
        print('------------------------')
        
        self.G_pruned = G_pruned
        self.waypoint_connections_pruned = connections_pruned  

    def make_graph_from_waypoints(self, max_distance=10, max_angle=45):
        '''
        Transform computed waypoints to a weighted, directed graph
        The nodes of the graph are self.waypoints
        The edges are calculated by iterating through all trajectories. 
        Edges are added between waypoints, when the trajectory has at most max_distance to the convex hulls of these waypoints and the difference
        in direction is at most max_angle
        '''
        print(f'Constructing maritime traffic network graph from waypoints and trajectories...')
        print(f'Progress:', end=' ', flush=True)
        start = time.time()  # start timer
        # create graph adjacency matrix
        n_clusters = len(self.waypoints)
        coord_dict = {}
        wps = self.waypoints.copy()
        wps.set_geometry('convex_hull', inplace=True)
        n_trajectories = len(self.significant_points.mmsi.unique())
        count = 0  # initialize a counter that keeps track of the progress
        percentage = 0  # percentage of progress
        # for each trajectory, find the distance to all waypoints
        for mmsi in self.significant_points.mmsi.unique():
            # find all intersections and close passages of waypoints
            trajectory = self.significant_points_trajectory.get_trajectory(mmsi)
            trajectory_segments = trajectory.to_line_gdf()
            distances = trajectory.distance(wps['convex_hull'])
            mask = distances < max_distance
            close_wps = wps[mask]
            # find temporal order  of waypoint passage
            passages = []  # initialize ordered list of waypoint passages per line segment
            for i in range(0, len(trajectory_segments)):
                segment = trajectory_segments.iloc[i]
                # distance of each segment to the selection of close waypoints
                distance_to_line = segment['geometry'].distance(close_wps['convex_hull'])  # distance between line segment and waypoint convex hull     
                distance_to_origin = segment['geometry'].boundary.geoms[0].distance(close_wps['geometry'])  # distance between first point of segment and waypoint centroids (needed for sorting)
                close_wps['distance_to_line'] = distance_to_line.tolist()
                close_wps['distance_to_origin'] = distance_to_origin.tolist()
                
                # angle between line segment and mean traffic direction in each waypoint
                WP_cog_before = close_wps['cog_before'] 
                WP_cog_after  = close_wps['cog_after']
                trajectory_cog = segment['direction']
                close_wps['angle_before'] = np.abs(WP_cog_before - trajectory_cog + 180) % 360 - 180
                close_wps['angle_after'] = np.abs(WP_cog_after - trajectory_cog + 180) % 360 - 180
                # the line segment is associated with the waypoint, when its distance and angle is less than a threshold
                mask = ((close_wps['distance_to_line']<max_distance) & 
                        (np.abs(close_wps['angle_before'])<max_angle) & 
                        (np.abs(close_wps['angle_after'])<max_angle))
                passed_wps = close_wps[mask]
                passed_wps.sort_values(by='distance_to_origin', inplace=True)
                passages.extend(passed_wps['clusterID'].tolist())
            
            # create edges between subsequent passed waypoints
            if len(passages) > 1:  # subset needs to contain at least 2 waypoints
                for i in range(0, len(passages)-1):
                    row = passages[i]
                    col = passages[i+1]
                    if row != col:  # no self loops
                        if (row, col) in coord_dict:
                            coord_dict[(row, col)] += 1  # increase the edge weight for each passage
                        else:
                            coord_dict[(row, col)] = 1  # create edge if it does not exist yet
            
            count += 1
            # report progress
            if count/n_trajectories > 0.1:
                count = 0
                percentage += 10
                print(f'{percentage}%...', end='', flush=True)

        # store adjacency matrix as sparse matrix in COO format
        row_indices, col_indices = zip(*coord_dict.keys())
        values = list(coord_dict.values())
        A = coo_matrix((values, (row_indices, col_indices)), shape=(n_clusters, n_clusters))

        # initialize directed graph from adjacency matrix
        G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)

        # add node features
        for i in range(0, len(self.waypoints)):
            node_id = self.waypoints.clusterID.iloc[i]
            G.nodes[node_id]['n_members'] = self.waypoints.n_members.iloc[i]
            G.nodes[node_id]['position'] = (self.waypoints.lon.iloc[i], self.waypoints.lat.iloc[i])  # !changed lat-lon to lon-lat for plotting
            G.nodes[node_id]['speed'] = self.waypoints.speed.iloc[i]
            G.nodes[node_id]['cog_before'] = self.waypoints.cog_before.iloc[i]
            G.nodes[node_id]['cog_after'] = self.waypoints.cog_after.iloc[i]
        
        # Construct a GeoDataFrame from graph edges (for plotting reasons)
        waypoints = self.waypoints.copy()
        waypoints.set_geometry('geometry', inplace=True, crs=self.crs)
        waypoint_connections = pd.DataFrame(columns=['from', 'to', 'geometry', 'direction', 'length', 'passages'])
        for orig, dest, weight in zip(A.row, A.col, A.data):
            # add linestring as edge
            p1 = waypoints[waypoints.clusterID == orig]['geometry']
            p2 = waypoints[waypoints.clusterID == dest]['geometry']
            edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
            length = edge.length
            # compute the orientation fo the edge (COG)
            p1 = Point(waypoints[waypoints.clusterID == orig].lon, waypoints[waypoints.clusterID == orig].lat)
            p2 = Point(waypoints[waypoints.clusterID == dest].lon, waypoints[waypoints.clusterID == dest].lat)
            direction = geometry_utils.calculate_initial_compass_bearing(p1, p2)
            line = pd.DataFrame([[orig, dest, edge, direction, length, weight]], 
                                columns=['from', 'to', 'geometry', 'direction', 'length', 'passages'])
            waypoint_connections = pd.concat([waypoint_connections, line])
        waypoint_connections = gpd.GeoDataFrame(waypoint_connections, geometry='geometry', crs=self.crs)

        # Add edge features
        for i in range(0, len(waypoint_connections)):
            orig = waypoint_connections['from'].iloc[i]
            dest = waypoint_connections['to'].iloc[i]
            G[orig][dest]['length'] = waypoint_connections['length'].iloc[i]
            G[orig][dest]['direction'] = waypoint_connections['direction'].iloc[i]
            G[orig][dest]['geometry'] = waypoint_connections['geometry'].iloc[i]
            G[orig][dest]['inverse_weight'] = 1/waypoint_connections['passages'].iloc[i]
       
        # report and save results
        print('Done!')
        print('------------------------')
        print(f'Unpruned Graph:')
        print(f'Number of nodes: {G.number_of_nodes()} ({nx.number_of_isolates(G)} isolated)')
        print(f'Number of edges: {G.number_of_edges()}')
        print(f'Network is (weakly) connected: {nx.is_weakly_connected(G)}')
        print('------------------------')
        self.G = G
        self.waypoint_connections = gpd.GeoDataFrame(waypoint_connections, geometry='geometry', crs=self.crs)
        
        end = time.time()
        print(f'Time elapsed: {(end-start)/60:.2f} minutes')      

    def trajectory_to_path(self, trajectory):
        '''
        find the best path along the graph for a given trajectory and evaluate goodness of fit
        :param trajectory: a single MovingPandas Trajectory object
        '''
        G = self.G_pruned.copy()
        waypoints = self.waypoints.copy()
        connections = self.waypoint_connections_pruned.copy()
        points = trajectory.to_point_gdf()
        mmsi = points.mmsi.unique()[0]
        #print('=======================')
        #print(mmsi)
        #print('=======================')
        
        ### GET potential START POINT ###
        orig_WP, idx_orig, dist_orig = geometry_utils.find_orig_WP(points, waypoints)
        
        ### GET potential END POINT ###
        dest_WP, idx_dest, dist_dest = geometry_utils.find_dest_WP(points, waypoints)
        #print('Potential start and end point:', orig_WP, dest_WP)

        ### GET ALL INTERSECTIONS between trajectory and waypoints
        passages = geometry_utils.find_WP_intersections(trajectory, waypoints)
        #print('Intersections found:', len(passages))
        
        # Distinguish three cases
        # passages is empty and orig != dest
        if ((len(passages) == 0) & (orig_WP != dest_WP)):
            if dist_orig < 100:
                passages.append(orig_WP)
            if dist_dest < 100:
                passages.append(dest_WP)
        # passages is empty and orig == dest --> nothing we can do here
        elif ((len(passages) == 0) & (orig_WP == dest_WP)):
            passages = []
        # found passages
        else:
            if ((passages[0] != orig_WP) & (dist_orig < 100)):
                passages.insert(0, orig_WP)
            else:
                orig_WP = passages[0]
                orig_WP_point = waypoints[waypoints.clusterID==orig_WP]['geometry'].item()
                idx_orig = np.argmin(orig_WP_point.distance(points.geometry))
            
            if ((passages[-1] != dest_WP) & (dist_dest < 100)):
                passages.append(dest_WP)
            else:
                dest_WP = passages[-1]
                dest_WP_point = waypoints[waypoints.clusterID==dest_WP]['geometry'].item()
                idx_dest = np.argmin(dest_WP_point.distance(points.geometry))
        
        if len(passages) >= 2:
            try:
                #print('Actual start and end point:', passages[0], passages[-1])
                # find edge sequence between each waypoint pair, that minimizes the distance between trajectory and edge sequence
                path = []
                #eval_distances = []
                for i in range(0, len(passages)-1):
                    #min_sequence = nx.shortest_path(G, passages[i], passages[i+1])
                    #edge_sequences = nx.all_simple_paths(G, passages[i], passages[i+1], cutoff=len(min_sequence)+1)
                    edge_sequences = nx.all_shortest_paths(G, passages[i], passages[i+1])
                    #print('=======================')
                    min_mean_distance = np.inf
                    for edge_sequence in edge_sequences:
                        # create a linestring from the edge sequence
                        multi_line = []
                        for j in range(0, len(edge_sequence)-1):
                            line = connections[(connections['from'] == edge_sequence[j]) & (connections['to'] == edge_sequence[j+1])].geometry.item()
                            multi_line.append(line)
                        multi_line = MultiLineString(multi_line)
                        multi_line = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                        # measure distance between the multi_line and the trajectory
                        WP1 = waypoints[waypoints.clusterID==edge_sequence[0]]['geometry'].item()
                        WP2 = waypoints[waypoints.clusterID==edge_sequence[-1]]['geometry'].item()
                        idx1 = np.argmin(WP1.distance(points.geometry))
                        idx2 = np.argmin(WP2.distance(points.geometry))
                        #print(idx1, idx2)
                        if idx2 < idx1:
                            temp = idx1
                            idx1 = idx2
                            idx2 = temp
                        eval_points = points.iloc[idx1:idx2+1]
                        distances = eval_points.distance(multi_line)
                        mean_distance = np.mean(distances)
                        #print(edge_sequence)
                        #print(mean_distance)
                        if mean_distance < min_mean_distance:
                            min_mean_distance = mean_distance
                            best_sequence = edge_sequence
                            #best_distances = distances
                    #eval_distances.append(best_distances.to_list())
                    path.append(best_sequence)
                    #print('----------------------')
                # delete duplicates from list
                flattened_path = [item for sublist in path for item in sublist]
                temp = [flattened_path[0]]
                for i in range(1, len(flattened_path)):
                    if flattened_path[i] != flattened_path[i-1]:
                        temp.append(flattened_path[i])
                path = temp
                message = 'success'
                #print(mmsi, nx.is_path(G, path))
                #print('Found path:', path)
        
                path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                for j in range(0, len(path)-1):
                    #print(path[j], path[j+1])
                    edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                    temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                    path_df = pd.concat([path_df, temp])
                path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
        
                ###########
                # evaluate goodness of fit
                ###########
                if idx_orig >= idx_dest:
                    idx_orig = 0
                    idx_dest = -1
                eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
                multi_line = MultiLineString(list(path_df.geometry))
                edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
                mean_dist = distances.mean()
                median_dist = distances.median()
                max_dist = distances.max()
                t1 = points.index[idx_orig]
                t2 = points.index[idx_dest]
                try:
                    percentage_covered = trajectory.get_linestring_between(t1, t2).length / trajectory.get_length()
                except:
                    percentage_covered = 1
                #percentage_covered = (len(eval_distances)-len(path)) / len(points)
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'mean_dist':mean_dist,
                                                   'median_dist':median_dist,
                                                   'max_dist':max_dist,
                                                   'distances':[distances.tolist()],
                                                   'fraction_covered':percentage_covered,
                                                   'message':message}
                                                 )
                #print(mmsi, ': success')
            
            except:
                if nx.has_path(G, passages[0], passages[-1]):
                    path = nx.shortest_path(G, passages[0], passages[-1])
                    message = 'attempt'
                    path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                    for j in range(0, len(path)-1):
                        #print(path[j], path[j+1])
                        edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                        temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                        path_df = pd.concat([path_df, temp])
                    path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
            
                    ###########
                    # evaluate goodness of fit
                    ###########
                    eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
                    multi_line = MultiLineString(list(path_df.geometry))
                    edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                    distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
                    mean_dist = distances.mean()
                    median_dist = distances.median()
                    max_dist = distances.max()
                    t1 = points.index[idx_orig]
                    t2 = points.index[idx_dest]
                    try:
                        percentage_covered = trajectory.get_linestring_between(t1, t2).length / trajectory.get_length()
                    except:
                        percentage_covered = 1
                    #percentage_covered = len(eval_points) / len(points)
                    evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                       'mean_dist':mean_dist,
                                                       'median_dist':median_dist,
                                                       'max_dist':max_dist,
                                                       'distances':[distances.tolist()],
                                                       'fraction_covered':percentage_covered,
                                                       'message':message}
                                                     )
                else:
                    message = 'no_path'
                    path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
                    evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                       'mean_dist':np.nan,
                                                       'median_dist':np.nan,
                                                       'max_dist':np.nan,
                                                       'distances':[np.nan],
                                                       'fraction_covered':0,
                                                       'message':message}
                                                     )
        
        else:
            #print('Not enough intersections found. Cannot map trajectory to graph...')
            message = 'no_intersects'
            #print(mmsi, ': failure')
            path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
            evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                               'mean_dist':np.nan,
                                               'median_dist':np.nan,
                                               'max_dist':np.nan,
                                               'distances':[np.nan],
                                               'fraction_covered':0,
                                               'message':message}
                                             )
        return path_df, evaluation_results
    
    def dijkstra_shortest_path(self, orig, dest, weight='inverse_weight'):
        '''
        outputs the shortest path in the network using Dijkstra's algorithm.
        :param orig: int, ID of the start waypoint
        :param dest: int, ID of the destination waypoint
        :param weight: string, name of the edge feature to be used as weight in Dijsktra's algorithm
        '''     
        try:
            # compute shortest path using dijsktra's algorithm (outputs a list of nodes)
            shortest_path = nx.shortest_path(self.G_pruned, orig, dest, weight=weight)
        except:
            print(f'Nodes {orig} and {dest} are not connected. Exiting...')
            return False
        else:
            # generate plottable GeoDataFrame from dijkstra path
            dijkstra_path_df = pd.DataFrame(columns=['orig', 'dest', 'geometry'])
            connections = self.waypoint_connections_pruned.copy()
            for j in range(0, len(shortest_path)-1):
                edge = connections[(connections['from'] == shortest_path[j]) & (connections['to'] == shortest_path[j+1])].geometry.item()
                temp = pd.DataFrame([[shortest_path[j], shortest_path[j+1], edge]], columns=['orig', 'dest', 'geometry'])
                dijkstra_path_df = pd.concat([dijkstra_path_df, temp])
            dijkstra_path_df = gpd.GeoDataFrame(dijkstra_path_df, geometry='geometry', crs=self.crs)
            return dijkstra_path_df
    
    def evaluate_graph(self, trajectories):
        '''
        given a selection of trajectories, compute evaluation metrics for the graph
        '''
        all_paths = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
        all_evaluation_results = pd.DataFrame(columns = ['mmsi', 'mean_dist', 'median_dist', 'max_dist', 'distances', 'message'])
        n_trajectories = len(trajectories)
        print(f'Evaluating graph on {n_trajectories} trajectories')
        print(f'Progress:', end=' ', flush=True)
        count = 0  # initialize a counter that keeps track of the progress
        percentage = 0  # percentage of progress
        for trajectory in trajectories:
            path, evaluation_results = self.trajectory_to_path(trajectory)
            all_paths = pd.concat([all_paths, path])
            all_evaluation_results = pd.concat([all_evaluation_results, evaluation_results])
            count += 1
            # report progress
            if count/n_trajectories > 0.1:
                count = 0
                percentage += 10
                print(f'{percentage}%...', end='', flush=True)
        print('Done!')

        # get percentages of success / attempt / orig_is_dest
        print('Success rates:')
        print((all_evaluation_results.groupby('message').count() / len(all_evaluation_results)).to_string())
        print('\n --------------------------- \n')
        
        # percentage of trajectories that could not be mapped properly
        nan_mask = all_evaluation_results.isna().any(axis=1)
        num_rows_with_nan = all_evaluation_results[nan_mask].shape[0]
        percentage_nan = num_rows_with_nan/len(all_evaluation_results)
        print(f'Fraction of NaN results: {percentage_nan:.3f}')
        print('\n --------------------------- \n')
        mean_fraction_covered = all_evaluation_results[~nan_mask]['fraction_covered'].mean()
        print(f'Mean fraction of each trajectory covered by the path on the graph: {mean_fraction_covered:.3f} \n')
        
        not_nan = all_evaluation_results[~nan_mask]
        # distances = not_nan[not_nan.message=='success']['distances'].tolist()
        distances = not_nan['distances'].tolist()
        distances = [item for sublist in distances for item in sublist]
        
        print(f'Mean distance      = {np.mean(distances):.2f} m')
        print(f'Median distance    = {np.median(distances):.2f} m')
        print(f'Standard deviation = {np.std(distances):.2f} m \n')
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        # Plot 1: Distance between each trajectory and edge sequence
        plot_evaluation = all_evaluation_results[~nan_mask]
        axes[0].boxplot(plot_evaluation.distances)
        axes[0].set_title('Distance between trajectory \n and edge sequence')
        axes[0].set_ylabel('Distance (m)')
        axes[0].set_xlabel(f'{n_trajectories} trajectories')
        axes[0].set_xticks([])

        # Plot 2: Distance between all trajectories and edge sequences (outlier cutoff at 2000m)
        axes[1].boxplot(distances)
        axes[1].set_title('Distance between all trajectories \n and edge sequences \n (outlier cutoff at 2000m)')
        axes[1].set_ylabel('Distance (m)')
        axes[1].set_ylim([0, 2000])

        # Plot 3: Distribution of distance
        axes[2].hist(distances, bins=np.arange(0, 2000, 20).tolist())
        axes[2].set_title('Distribution of distance')
        axes[2].set_xlabel('Distance (m)')
        
        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.show()
        
        summary = {'Mean distance':np.mean(distances),
                   'Median distance':np.median(distances),
                   'Standard deviation':np.std(distances),
                   'Fraction NaN':percentage_nan,
                   'Fraction success':len(all_evaluation_results[all_evaluation_results.message=='success']) / len(all_evaluation_results),
                   'mean fraction covered': mean_fraction_covered}

        all_paths = gpd.GeoDataFrame(all_paths, geometry='geometry', crs=self.crs)
        return all_paths, all_evaluation_results, summary, fig
    
    def map_waypoints(self, detailed_plot=False, center=[59, 5]):
        # plotting
        if detailed_plot:
            columns = ['geometry', 'mmsi']  # columns to be plotted
            # plot simplified trajectories
            map = self.trajectories.to_traj_gdf()[columns].explore(column='mmsi', name='Simplified trajectories', 
                                                                      style_kwds={'weight':1, 'color':'black', 'opacity':0.5}, 
                                                                      legend=False)
            # plot significant turning points with their cluster ID
            map = self.significant_points[['clusterID', 'geometry']].explore(m=map, name='all waypoints with cluster ID', 
                                                                                legend=False,
                                                                                marker_kwds={'radius':2},
                                                                                style_kwds={'opacity':0.2})
        else:
            # plot basemap
            map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=8)
            # plot traffic as raster overlay
            map = visualize.traffic_raster_overlay(self.gdf.to_crs(4326), map)
        
        # plot cluster centroids and their convex hull
        cluster_centroids = self.waypoints.copy()
        cluster_centroids.to_crs(4326, inplace=True)
        columns_points = ['clusterID', 'geometry', 'speed', 'cog_before', 'cog_after', 'n_members']  # columns to plot
        columns_hull = ['clusterID', 'convex_hull', 'speed', 'cog_before', 'cog_after', 'n_members']  # columns to plot
        
        # plot eastbound cluster centroids
        eastbound = cluster_centroids[(cluster_centroids.cog_after < 180) & (cluster_centroids.speed >= 2.0)]
        eastbound.set_geometry('geometry', inplace=True)
        map = eastbound[columns_points].explore(m=map, name='cluster centroids (eastbound)', legend=False,
                                                marker_kwds={'radius':3},
                                                style_kwds={'color':'green', 'fillColor':'green', 'fillOpacity':1})
        eastbound.set_geometry('convex_hull', inplace=True, crs=self.crs)
        map = eastbound[columns_hull].explore(m=map, name='cluster convex hulls (eastbound)', legend=False,
                                              style_kwds={'color':'green', 'fillColor':'green', 'fillOpacity':0.2})
        
        # plot westbound cluster centroids
        westbound = cluster_centroids[(cluster_centroids.cog_after >= 180) & (cluster_centroids.speed >= 2.0)]
        westbound.set_geometry('geometry', inplace=True)
        map = westbound[columns_points].explore(m=map, name='cluster centroids (westbound)', legend=False,
                                                marker_kwds={'radius':3},
                                                style_kwds={'color':'red', 'fillColor':'red', 'fillOpacity':1})
        westbound.set_geometry('convex_hull', inplace=True, crs=self.crs)
        map = westbound[columns_hull].explore(m=map, name='cluster convex hulls (westbound)', legend=False,
                                              style_kwds={'color':'red', 'fillColor':'red', 'fillOpacity':0.2})
        
        # plot stop cluster centroids
        stops = cluster_centroids[cluster_centroids.speed < 2.0]
        if len(stops) > 0:
            stops.set_geometry('geometry', inplace=True)
            map = stops[columns_points].explore(m=map, name='cluster centroids (stops)', legend=False,
                                                marker_kwds={'radius':3},
                                                style_kwds={'color':'blue', 'fillColor':'blue', 'fillOpacity':1})
            stops.set_geometry('convex_hull', inplace=True, crs=self.crs)
            map = stops[columns_hull].explore(m=map, name='cluster convex hulls (stops)', legend=False,
                                              style_kwds={'color':'blue', 'fillColor':'blue', 'fillOpacity':0.2})
        #folium.LayerControl().add_to(map)

        return map

    def map_graph(self, pruned=False):
        '''
        Visualization function to map the maritime traffic network graph
        '''
        # basemap with waypoints and traffic
        map = self.map_waypoints()

        # add connections
        if pruned:
            connections = self.waypoint_connections_pruned.copy()
        else:
            connections = self.waypoint_connections.copy()
        eastbound = connections[(connections.direction < 180)]
        westbound = connections[(connections.direction >= 180)]
        map = westbound.explore(m=map, name='westbound graph edges', legend=False,
                                style_kwds={'weight':2, 'color':'red', 'opacity':0.7})
        map = eastbound.explore(m=map, name='eastbound graph edges', legend=False,
                                style_kwds={'weight':2, 'color':'green', 'opacity':0.7})
        return map
    
    def plot_graph_canvas(self):
        '''
        Plot the maritime traffic network graph on a white canvas
        '''
        G = self.G
        # Create a dictionary mapping nodes to their positions
        node_positions = {node: G.nodes[node]['position'] for node in G.nodes}
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 4]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 4]
        
        # Create the network plot
        plt.figure(figsize=(10, 10))
        #nx.draw(G, pos=node_positions, with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_color='black', font_weight='bold')
        # nodes
        nx.draw_networkx_nodes(G, pos=node_positions, node_size=200, node_color='skyblue')
        # edges
        nx.draw_networkx_edges(G, pos=node_positions, edgelist=elarge, width=3)
        nx.draw_networkx_edges(G, pos=node_positions, edgelist=esmall, width=1, alpha=0.5)
        
        # node labels
        nx.draw_networkx_labels(G, pos=node_positions, font_size=8, font_family="sans-serif")
        # edge weight labels
        #edge_labels = nx.get_edge_attributes(G, "weight")
        #nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_labels)
        
        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        # Show the plot
        plt.show()

    def LEGACY_make_graph_from_waypoints(self, min_passages=3):
        '''
        LEGACY function
        Transform computed waypoints to a weighted, directed graph
        The nodes of the graph are self.waypoints
        The edges are calculated by iterating through all significant points in each trajectory. 
        The significant points have been assigned a clusterID. Edges are added between pairwise clusters that follow each other 
        in the significant points dataframe.
        Example: The clusterID column of the significant point dataframe looks like this:
                 1 1 1 4 4 5 5 7
                 The algorithm extract the following edges from that:
                 1-4, 4-5, 5-7
        Weakness: Clusters that are intersected by the actual trajectory are not added as edges. The function aggregate_edges()
        is built to rectify that, but is slow and does not catch all clusters.
        THIS ALGORITHM NEEDS TO BE IMPROVED
        '''     
        print(f'Constructing maritime traffic network graph from waypoints and trajectories...')
        start = time.time()  # start timer
        # create graph adjacency matrix
        n_clusters = len(self.waypoints)
        coord_dict = {}
        # for each trajectory, increase the weight of the adjacency matrix between two nodes
        for mmsi in self.significant_points.mmsi.unique():
            subset = self.significant_points[self.significant_points.mmsi == mmsi]
            subset = subset[subset.clusterID >=0]  # remove outliers
            if len(subset) > 1:  # subset needs to contain at least 2 waypoints
                for i in range(0, len(subset)-1):
                    row = subset.clusterID.iloc[i]
                    col = subset.clusterID.iloc[i+1]
                    if row != col:  # no self loops
                        if (row, col) in coord_dict:
                            coord_dict[(row, col)] += 1  # increase the edge weight for each passage
                        else:
                            coord_dict[(row, col)] = 1  # create edge if it does not exist yet
        
        # store adjacency matrix as sparse matrix in COO format
        row_indices, col_indices = zip(*coord_dict.keys())
        values = list(coord_dict.values())
        A = coo_matrix((values, (row_indices, col_indices)), shape=(n_clusters, n_clusters))

        # Construct a GeoDataFrame from graph edges
        waypoints = self.waypoints
        waypoints.set_geometry('geometry', inplace=True, crs=self.crs)
        waypoint_connections = pd.DataFrame(columns=['from', 'to', 'geometry', 'direction', 'passages'])
        for orig, dest, weight in zip(A.row, A.col, A.data):
            # add linestring as edge
            p1 = waypoints[waypoints.clusterID == orig].geometry
            p2 = waypoints[waypoints.clusterID == dest].geometry
            edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
            # compute the orientation fo the edge (COG)
            p1 = Point(waypoints[waypoints.clusterID == orig].lon, waypoints[waypoints.clusterID == orig].lat)
            p2 = Point(waypoints[waypoints.clusterID == dest].lon, waypoints[waypoints.clusterID == dest].lat)
            direction = geometry_utils.calculate_initial_compass_bearing(p1, p2)
            line = pd.DataFrame([[orig, dest, edge, direction, weight]], 
                                columns=['from', 'to', 'geometry', 'direction', 'passages'])
            waypoint_connections = pd.concat([waypoint_connections, line])

        # Aggregate edges recursively
        # each edge that intersects the convex hull of another waypoint is divided in segments
        # the segments are added to the adjacency matrix and the original edge is deleted
        A_refined, waypoint_connections_refined, flag = geometry_utils.aggregate_edges(waypoints, waypoint_connections)
        while flag:
            A_refined, waypoint_connections_refined, flag = geometry_utils.aggregate_edges(waypoints, waypoint_connections_refined)
        
        # Construct a GeoDataFrame from graph edges
        waypoints = self.waypoints
        waypoints.set_geometry('geometry', inplace=True, crs=self.crs)
        waypoint_connections = pd.DataFrame(columns=['from', 'to', 'geometry', 'direction', 'passages'])
        for orig, dest, weight in zip(A_refined.row, A_refined.col, A_refined.data):
            # add linestring as edge
            p1 = waypoints[waypoints.clusterID == orig].geometry
            p2 = waypoints[waypoints.clusterID == dest].geometry
            edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
            # compute the orientation fo the edge (COG)
            p1 = Point(waypoints[waypoints.clusterID == orig].lon, waypoints[waypoints.clusterID == orig].lat)
            p2 = Point(waypoints[waypoints.clusterID == dest].lon, waypoints[waypoints.clusterID == dest].lat)
            direction = geometry_utils.calculate_initial_compass_bearing(p1, p2)
            line = pd.DataFrame([[orig, dest, edge, direction, weight]], 
                                columns=['from', 'to', 'geometry', 'direction', 'passages'])
            waypoint_connections = pd.concat([waypoint_connections, line])
        
        # initialize directed graph from adjacency matrix
        G = nx.from_scipy_sparse_array(A_refined, create_using=nx.DiGraph)

        # add node features
        for i in range(0, len(self.waypoints)):
            node_id = self.waypoints.clusterID.iloc[i]
            G.nodes[node_id]['n_members'] = self.waypoints.n_members.iloc[i]
            G.nodes[node_id]['position'] = (self.waypoints.lon.iloc[i], self.waypoints.lat.iloc[i])  # !changed lat-lon to lon-lat for plotting
            G.nodes[node_id]['speed'] = self.waypoints.speed.iloc[i]
            G.nodes[node_id]['cog_before'] = self.waypoints.cog_before.iloc[i]
            G.nodes[node_id]['cog_after'] = self.waypoints.cog_after.iloc[i]

        
        # report and save results
        print('------------------------')
        print(f'Unpruned Graph:')
        print(f'Number of nodes: {G.number_of_nodes()}')
        print(f'Number of edges: {G.number_of_edges()}')
        print('------------------------')
        self.G = G
        self.waypoint_connections = gpd.GeoDataFrame(waypoint_connections, geometry='geometry', crs=self.crs)

        # Prune network
        self.prune_graph(min_passages)
        
        end = time.time()  # end timer
        print(f'Time elapsed: {(end-start)/60:.2f} minutes')

    def LEGACY_trajectory_to_path(self, trajectory):
        '''
        find the best path along the graph for a given trajectory and evaluate goodness of fit
        :param trajectory: a single MovingPandas Trajectory object
        '''
        G = self.G_pruned.copy()
        waypoints = self.waypoints.copy()
        connections = self.waypoint_connections_pruned.copy()
        points = trajectory.to_point_gdf()
        mmsi = points.mmsi.unique()[0]
        #print('=======================')
        #print(mmsi)
        #print('=======================')
        
        ### GET START POINT ###
        orig_WP, idx_orig, _ = geometry_utils.find_orig_WP(points, waypoints)
        
        ### GET END POINT ###
        dest_WP, idx_dest, _ = geometry_utils.find_dest_WP(points, waypoints)
        #print(orig_WP, dest_WP)
        
        try:
        # find all waypoints intersected by the trajectory
            #print('try')
            passages = geometry_utils.find_WP_intersections(trajectory, waypoints)
            if passages[0] != orig_WP:
                passages.insert(0, orig_WP)
            if passages[-1] != dest_WP:
                passages.append(dest_WP)
            if len(passages)<2:
                print('not enough intersections found')
                raise Exception
            #print(passages)
            # find edge sequence between each waypoint pair, that minimizes the distance between trajectory and edge sequence
            path = []
            for i in range(0, len(passages)-1):
                #min_sequence = nx.shortest_path(G, passages[i], passages[i+1])
                #edge_sequences = nx.all_simple_paths(G, passages[i], passages[i+1], cutoff=len(min_sequence)+1)
                edge_sequences = nx.all_shortest_paths(G, passages[i], passages[i+1])
                min_mean_distance = np.inf
                for edge_sequence in edge_sequences:
                    # create a linestring from the edge sequence
                    multi_line = []
                    for j in range(0, len(edge_sequence)-1):
                        line = connections[(connections['from'] == edge_sequence[j]) & (connections['to'] == edge_sequence[j+1])].geometry.item()
                        multi_line.append(line)
                    multi_line = MultiLineString(multi_line)
                    multi_line = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                    # measure distance between the multi_line and the trajectory
                    WP1 = waypoints[waypoints.clusterID==edge_sequence[j]]['geometry'].item()
                    WP2 = waypoints[waypoints.clusterID==edge_sequence[j+1]]['geometry'].item()
                    idx1 = np.argmin(WP1.distance(points.geometry))
                    idx2 = np.argmin(WP2.distance(points.geometry))
                    if idx2 < idx1:
                        temp = idx1
                        idx1 = idx2
                        idx2 = temp
                    eval_points = points.iloc[idx1:idx2+1]
                    distances = eval_points.distance(multi_line)
                    mean_distance = np.mean(distances)
                    #print(edge_sequence)
                    #print(mean_distance)
                    if mean_distance < min_mean_distance:
                        min_mean_distance = mean_distance
                        best_sequence = edge_sequence
                path.append(best_sequence)
                #print('----------------------')
            flattened_path = [item for sublist in path for item in sublist]
            path = list(dict.fromkeys(flattened_path))
            message = 'success'
            #print('Found path:', path)
    
            path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
            for j in range(0, len(path)-1):
                #print(path[j], path[j+1])
                edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                path_df = pd.concat([path_df, temp])
            path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
    
            ###########
            # evaluate goodness of fit
            ###########
            eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
            multi_line = MultiLineString(list(path_df.geometry))
            edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
            distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
            mean_dist = distances.mean()
            median_dist = distances.median()
            max_dist = distances.max()
            evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                               'mean_dist':mean_dist,
                                               'median_dist':median_dist,
                                               'max_dist':max_dist,
                                               'distances':[distances.tolist()],
                                               'message':message}
                                             )
            #print(mmsi, ': success')
        
        except:
            if orig_WP == dest_WP:
                #print('origin is destination. Exiting...')
                message = 'orig_is_dest'
                path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'mean_dist':np.nan,
                                                   'median_dist':np.nan,
                                                   'max_dist':np.nan,
                                                   'distances':[np.nan],
                                                   'message':message}
                                                 )
                #print(mmsi, ': orig_is_dest (no path)')
            
            else:
                try:
                    #print('Attemptin shortest path method')
                    path = nx.shortest_path(G, orig_WP, dest_WP)
                    message = 'attempt'
                    path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                    for j in range(0, len(path)-1):
                        edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                        temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                        path_df = pd.concat([path_df, temp])
                    path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
                    ###########
                    # evaluate goodness of fit
                    ###########
                    eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
                    multi_line = MultiLineString(list(path_df.geometry))
                    edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                    distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
                    mean_dist = distances.mean()
                    median_dist = distances.median()
                    max_dist = distances.max()
                    evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                       'mean_dist':mean_dist,
                                                       'median_dist':median_dist,
                                                       'max_dist':max_dist,
                                                       'distances':[distances.tolist()],
                                                       'message':message}
                                                     )
                    #print(mmsi, ': attempt')
                except:
                    message = 'failure'
                    #print(mmsi, ': failure')
                    path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
                    evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                       'mean_dist':np.nan,
                                                       'median_dist':np.nan,
                                                       'max_dist':np.nan,
                                                       'distances':[np.nan],
                                                       'message':message}
                                                     )
        
        return path_df, evaluation_results