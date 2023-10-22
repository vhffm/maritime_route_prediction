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
import networkx as nx
import time
import folium
import warnings
import sys

# add paths for modules
sys.path.append('../visualization')
sys.path.append('../features')
print(sys.path)

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
            metric_params = {'V':V}   # np.diag([0.01, 0.01, 1e6, 1e6])  # mahalanobis distance parameter matrix
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

    def prune_graph(self, min_passages):
        '''
        prunes the maritime traffic graph to only contain edges with more passages than min_passages
        '''
        A = nx.to_scipy_sparse_array(self.G, format='coo')
        mask = A.data >= min_passages
        A_pruned = coo_matrix((A.data[mask], (A.row[mask], A.col[mask])), shape=A.shape)
        G_pruned = nx.from_scipy_sparse_array(A_pruned, create_using=nx.DiGraph)
        G_pruned.nodes = self.G.nodes
        self.waypoint_connections_pruned = self.waypoint_connections[self.waypoint_connections.passages >= min_passages]
        # add edge features
        for i in range(0, len(self.waypoint_connections_pruned)):
            orig = self.waypoint_connections_pruned['from'].iloc[i]
            dest = self.waypoint_connections_pruned['to'].iloc[i]
            G_pruned[orig][dest]['length'] = self.waypoint_connections_pruned['length'].iloc[i]
            G_pruned[orig][dest]['direction'] = self.waypoint_connections_pruned['direction'].iloc[i]
            G_pruned[orig][dest]['geometry'] = self.waypoint_connections_pruned['geometry'].iloc[i]
            G_pruned[orig][dest]['inverse_weight'] = 1/self.waypoint_connections['passages'].iloc[i]
        self.G_pruned = G_pruned
        print('------------------------')
        print(f'Pruned Graph:')
        print(f'Number of nodes: {G_pruned.number_of_nodes()} ({nx.number_of_isolates(G_pruned)} isolated)')
        print(f'Number of edges: {G_pruned.number_of_edges()}')
        print('------------------------')
    

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
            distances = trajectory.distance(wps.convex_hull)
            mask = distances < max_distance
            close_wps = wps[mask]
            # find temporal order  of waypoint passage
            passages = []  # initialize ordered list of waypoint passages per line segment
            for i in range(0, len(trajectory_segments)):
                segment = trajectory_segments.iloc[i]
                # distance of each segment to the selection of close waypoints
                distance_to_line = segment['geometry'].distance(close_wps.convex_hull)  # distance between line segment and waypoint convex hull     
                distance_to_origin = segment['geometry'].boundary.geoms[0].distance(close_wps.geometry)  # distance between first point of segment and waypoint centroids (needed for sorting)
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

                '''
                # angle between line segment and mean traffic direction in each waypoint
                WP_cog = geometry_utils.compass_mean(close_wps['cog_before'], close_wps['cog_after'])
                close_wps['angle'] = np.abs(WP_cog - segment['direction'] + 180) % 360 - 180
                # the line segment is associated with the waypoint, when its distance and angle is less than a threshold
                passed_wps = close_wps[(close_wps['distance_to_line']<max_distance) & (close_wps['angle']<max_angle)]
                passed_wps.sort_values(by='distance_to_origin', inplace=True)
                passages.extend(passed_wps['clusterID'].tolist())
                '''
            
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
            p1 = waypoints[waypoints.clusterID == orig].geometry
            p2 = waypoints[waypoints.clusterID == dest].geometry
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
        
        ### GET START POINT ###
        # map trajectory start point to the closest waypoint available
        orig = points.iloc[0].geometry  # get trajectory start point
        # find out if trajectory starts in a stop point
        try:
            orig_speed = points.iloc[0:5].speed.mean()
            if orig_speed < 2:
                orig_cog = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[0].lon, points.iloc[0].lat), 
                                                                            Point(points.iloc[40].lon, points.iloc[40].lat)) # get initial cog
                angle = (orig_cog - waypoints.cog_after + 180) % 360 - 180
                #mask = ((waypoints.speed < 2) & (np.abs(angle) < 45))
                mask = np.abs(angle) < 45
            else:
                orig_cog = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[0].lon, points.iloc[0].lat), 
                                                                            Point(points.iloc[9].lon, points.iloc[9].lat)) # get initial cog
                angle = (orig_cog - waypoints.cog_after + 180) % 360 - 180
                mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
        except:
            orig_speed = points.iloc[0:2].speed.mean()
            orig_cog = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[0].lon, points.iloc[0].lat), 
                                                                            Point(points.iloc[1].lon, points.iloc[1].lat)) # get initial cog
            angle = (orig_cog - waypoints.cog_after + 180) % 360 - 180
            mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
        distances = orig.distance(waypoints[mask].geometry)
        masked_idx = np.argmin(distances)
        orig_WP = waypoints[mask]['clusterID'].iloc[masked_idx]
        # find trajectory point that is closest to the centroid of the first waypoint
        # this is where we start measuring
        orig_WP_point = waypoints[waypoints.clusterID==orig_WP]['geometry'].item()
        idx_orig = np.argmin(orig_WP_point.distance(points.geometry))
        # Now we have found the first waypoint and the first point on the trajectory closest to that waypoint
        
        ### GET END POINT ###
        # map trajectory end point to the closest waypoint available
        dest = points.iloc[-1].geometry  # get end point
        try:
            dest_speed = points.iloc[-5:-1].speed.mean()
            if dest_speed < 2:
                mask = (waypoints.speed < 2)
            else:
                dest_cog = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[-10].lon, points.iloc[-10].lat), 
                                                                            Point(points.iloc[-1].lon, points.iloc[-1].lat)) # get initial cog
                angle = (dest_cog - waypoints.cog_before + 180) % 360 - 180
                mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
        except:
            dest_speed = points.iloc[-3:-1].speed.mean()
            dest_cog = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[-2].lon, points.iloc[-2].lat), 
                                                                        Point(points.iloc[-1].lon, points.iloc[-1].lat)) # get initial cog
            angle = (dest_cog - waypoints.cog_before + 180) % 360 - 180
            mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
        distances = dest.distance(waypoints[mask].geometry)
        masked_idx = np.argmin(distances)
        dest_WP = waypoints[mask]['clusterID'].iloc[masked_idx]
        # find trajectory point that is closest to the centroid of the last waypoint
        # this is where we end measuring
        dest_WP_point = waypoints[waypoints.clusterID==dest_WP]['geometry'].item()
        idx_dest = np.argmin(dest_WP_point.distance(points.geometry))
        # Now we have found the last waypoint and the last point on the trajectory closest to that waypoint
        #print(f'start: {orig_WP}, end: {dest_WP}')
        #print(orig_speed, orig_cog, dest_speed, dest_cog)
        if orig_WP == dest_WP:
            print('Origin = Destination')
            message = 'orig_is_dest'
        #### Find path between origin and destination
        # find a path from start to end waypoint
        path = [orig_WP]
        current_WP = orig_WP
        while path[-1] != dest_WP:
            # Check if a next path segment is available
            if len(list(G.neighbors(current_WP))) > 0:
                next_WP = current_WP  # if we don't find a next waypoint, the algorithm will terminate
                # check if there are any intersections between candidate waypoints and the trajectory
                candidate_WPs = G.neighbors(current_WP)
                candidate_WP_hulls = waypoints[waypoints.clusterID.isin(candidate_WPs)]['convex_hull']
                intersects = trajectory.distance(candidate_WP_hulls)<50
                # if we found waypoint intersections we look for the next waypoint among them
                if np.sum(intersects)>0:
                    #print('Waypoint intersections found. Finding closest neighbor...')
                    shortest_edge_length = np.inf
                    # try to find closest waypoint that is intersected by the trajectory and matches in direction
                    for candidate_WP in G.neighbors(current_WP):
                        #print(f'Candidate: {candidate_WP}')
                        # does the trajectory intersect the convex hull of the waypoint candidate?
                        candidate_WP_hull = waypoints[waypoints.clusterID==candidate_WP]['convex_hull'].item()
                        intersects = trajectory.distance(candidate_WP_hull)<50
                        if intersects:
                            edge_length = G[current_WP][candidate_WP]['length']
                        
                            # find trajectory point closest to current waypoint
                            current_WP_point = waypoints[waypoints.clusterID==current_WP]['geometry'].item()  # get coordinates of waypoint centroid
                            distances = points.distance(current_WP_point)  
                            p1_idx = np.argmin(distances)
                        
                            # find trajectory point closest to candidate waypoint
                            candidate_WP_point = waypoints[waypoints.clusterID==candidate_WP]['geometry'].item()  # get coordinates of waypoint centroid
                            distances = points.iloc[p1_idx:].distance(candidate_WP_point)  
                            p2_idx = np.argmin(distances) + p1_idx
                            if (p2_idx - p1_idx) <= 0:
                                #print('going back is not allowed')
                                continue
        
                            # does the COG of trajectory and waypoint candidate match?
                            # compute trajectory COG close to candidate waypoint
                            if p2_idx >= 5:
                                p2_cog_before = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[p2_idx-5].lon, points.iloc[p2_idx-5].lat), 
                                                                                                 Point(points.iloc[p2_idx].lon, points.iloc[p2_idx].lat))
                            else:
                                p2_cog_before = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[p2_idx-1].lon, points.iloc[p2_idx-1].lat), 
                                                                                                 Point(points.iloc[p2_idx].lon, points.iloc[p2_idx].lat))
                            # caution at the end of the trajectory not to run out of bounds
                            if len(points)-p2_idx > 5:
                                p2_cog_after = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[p2_idx].lon, points.iloc[p2_idx].lat), 
                                                                                            Point(points.iloc[p2_idx+5].lon, points.iloc[p2_idx+5].lat))
                            else:
                                p2_cog_after = p2_cog_before
                        
                            # get candidate waypoint traffic direction both before and after waypoint passage
                            candidate_WP_cog_before = waypoints[waypoints.clusterID==candidate_WP]['cog_before'].item()
                            candidate_WP_cog_after = waypoints[waypoints.clusterID==candidate_WP]['cog_after'].item()
                        
                            # compute angle between trajectory and waypoint candidate
                            angle_before = (p2_cog_before - candidate_WP_cog_before + 180) % 360 - 180
                            angle_after = (p2_cog_after - candidate_WP_cog_after + 180) % 360 - 180
                            #print(intersects, edge_length)
                            #print(f'trajectory COG: {p2_cog_before:.2f}, WP COG before: {candidate_WP_cog_before:.2f}, angle: {angle_before:.2f}')
                            #print(f'trajectory COG: {p2_cog_after:.2f}, WP COG after: {candidate_WP_cog_after:.2f}, angle: {angle_after:.2f}')
                            if ((edge_length < shortest_edge_length) & ((np.abs(angle_before)<30 or np.abs(angle_after)<30))):
                                next_WP = candidate_WP
                                shortest_edge_length = edge_length
                else:
                    #print('No intersections found. Finding neighbor that minimises distance between trajectory and edge...')
                    shortest_mean_distance_intersects = np.inf
                    shortest_mean_distance_non_intersects = np.inf
                    flag = False
                    for candidate_WP in G.neighbors(current_WP):
                        #print(f'Candidate: {candidate_WP}')
                        # does the candidate have any neighbors that lead back to the trajectory?
                        candidate_WP_hulls = waypoints[waypoints.clusterID.isin(G.neighbors(candidate_WP))]['convex_hull']
                        intersects = trajectory.distance(candidate_WP_hulls)<50
        
                        # calculate mean distance between the trajectory and the edge between current and candidate waypoint
                        # get edge between current and candidate waypoint
                        edge = connections[(connections['from'] == current_WP) & (connections['to'] == candidate_WP)].geometry.item()
                        # get trajectory segment between the two waypoints
                        # find trajectory point closest to current waypoint
                        current_WP_point = waypoints[waypoints.clusterID==current_WP]['geometry'].item()  # get coordinates of waypoint centroid
                        distances = points.distance(current_WP_point)  
                        p1_idx = np.argmin(distances)
                        # find trajectory point closest to candidate waypoint
                        candidate_WP_point = waypoints[waypoints.clusterID==candidate_WP]['geometry'].item()  # get coordinates of waypoint centroid
                        distances = points.iloc[p1_idx:].distance(candidate_WP_point)  
                        p2_idx = np.argmin(distances) + p1_idx
                        if (p2_idx - p1_idx) <= 0:
                            #print('going back is not allowed')
                            continue
                        # compute mean distance between trajectory segment and edge
                        mean_distance = points[p1_idx:p2_idx].distance(edge).mean()
                        # compute trajectory COG close to candidate waypoint
                        if p2_idx >= 5:
                            p2_cog_before = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[p2_idx-5].lon, points.iloc[p2_idx-5].lat), 
                                                                                             Point(points.iloc[p2_idx].lon, points.iloc[p2_idx].lat))
                        else:
                            p2_cog_before = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[p2_idx-1].lon, points.iloc[p2_idx-1].lat), 
                                                                                             Point(points.iloc[p2_idx].lon, points.iloc[p2_idx].lat))
                        # caution at the end of the trajectory not to run out of bounds
                        if len(points)-p2_idx > 5:
                            p2_cog_after = geometry_utils.calculate_initial_compass_bearing(Point(points.iloc[p2_idx].lon, points.iloc[p2_idx].lat), 
                                                                                            Point(points.iloc[p2_idx+5].lon, points.iloc[p2_idx+5].lat))
                        else:
                            p2_cog_after = p2_cog_before
                        # get candidate waypoint traffic direction both before and after waypoint passage
                        candidate_WP_cog_before = waypoints[waypoints.clusterID==candidate_WP]['cog_before'].item()
                        candidate_WP_cog_after = waypoints[waypoints.clusterID==candidate_WP]['cog_after'].item()
                        angle_before = (p2_cog_before - candidate_WP_cog_before + 180) % 360 - 180
                        angle_after = (p2_cog_after - candidate_WP_cog_after + 180) % 360 - 180
        
                        if np.sum(intersects)>0:
                            flag=True  # found a point that leads back to path
                            if ((mean_distance < shortest_mean_distance_intersects) & ((np.abs(angle_before)<90 or np.abs(angle_after)<90))):
                                next_WP = candidate_WP
                                shortest_mean_distance_intersects = mean_distance
                        else:
                            if ((mean_distance < shortest_mean_distance_non_intersects) & ((np.abs(angle_before)<90 or np.abs(angle_after)<90)) & (flag==False)):
                                next_WP = candidate_WP
                                shortest_mean_distance_non_intersects = mean_distance
                    
                if next_WP == current_WP:
                    print('Got stuck. Terminating early...')
                    message = 'early_termination'
                    break
                else:
                    path.append(next_WP) # append next waypoint to path
                    G.remove_node(current_WP)  # drop current waypoint from graph to avoid going in circles
                    current_WP = next_WP
                    message = 'success'
                #print(f'Found next waypoint: {next_WP}')
                #print('==============================================')
            else:
                print('No more adjacent waypoints. Terminating...')
                message = 'no_path'
                break
            
        # make trajectory from path
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
        
        return path_df, evaluation_results      

    def trajectory_to_path2(self, trajectory):
        '''
        find the best path along the graph for a given trajectory and evaluate goodness of fit
        :param trajectory: a single MovingPandas Trajectory object
        '''
        G = self.G_pruned.copy()
        waypoints = self.waypoints.copy()
        connections = self.waypoint_connections_pruned.copy()
        points = trajectory.to_point_gdf()
        mmsi = points.mmsi.unique()[0]
        
        ### GET START POINT ###
        orig_WP, idx_orig = geometry_utils.find_orig_WP(points, waypoints)
        
        ### GET END POINT ###
        dest_WP, idx_dest = geometry_utils.find_dest_WP(points, waypoints)
        
        if orig_WP == dest_WP:
            print('Origin = Destination')
            message = 'orig_is_dest'
            path = [orig_WP, dest_WP]
        else:
            # find all waypoints intersected by the trajectory
            passages = geometry_utils.find_WP_intersections(trajectory, waypoints)
            if passages[0] != orig_WP:
                passages.insert(0, orig_WP)
            if passages[-1] != dest_WP:
                passages.append(dest_WP)
            # find edge sequence between each waypoint pair, that minimizes the distance between trajectory and edge sequence
            path = []
            for i in range(0, len(passages)-1):
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
        
        #print(path)
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