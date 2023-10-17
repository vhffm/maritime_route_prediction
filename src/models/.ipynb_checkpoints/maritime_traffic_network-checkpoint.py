import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from scipy.sparse import coo_matrix
from shapely.geometry import Point, LineString
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
        self.trajectories = mpd.TrajectoryCollection(self.gdf, traj_id_col='mmsi', obj_id_col='mmsi')
        self.significant_points_trajectory = []
        self.significant_points = []
        self.waypoints = []
        self.waypoint_connections = []
        self.waypoint_connections_unpruned = []
        self.G = []
        self.G_unpruned = []

    def get_trajectories_info(self):
        print(f'Number of AIS messages: {len(self.gdf)}')
        print(f'Number of trajectories: {len(self.trajectories)}')
        print(f'Coordinate Reference System (CRS): {self.gdf.crs}')

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
        self.G_pruned = G_pruned
        self.waypoint_connections_pruned = self.waypoint_connections[self.waypoint_connections.passages >= min_passages]
        print(f'Pruned Graph:')
        print(f'Number of nodes: {G_pruned.number_of_nodes()}')
        print(f'Number of edges: {G_pruned.number_of_edges()}')
    

    def make_graph_from_waypoints(self, max_distance=100, max_angle=45):
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
            passages = []  # initialize ordered lit of waypoint passages per line segment
            for i in range(0, len(trajectory_segments)):
                segment = trajectory_segments.iloc[i]
                # distance of each segment to the selection of close waypoints
                distance_to_line = segment['geometry'].distance(close_wps.convex_hull)  # distance between line segment and waypoint convex hull     
                distance_to_origin = segment['geometry'].boundary.geoms[0].distance(close_wps.geometry)  # distance between first point of segment and waypoint centroids (needed for sorting)
                close_wps['distance_to_line'] = distance_to_line.tolist()
                close_wps['distance_to_origin'] = distance_to_origin.tolist()
                # angle between line segment and mean traffic direction in each waypoint
                close_wps['angle'] = np.abs(geometry_utils.compass_mean(close_wps['cog_before'], close_wps['cog_after']) - segment['direction']) 
                # the line segment is associated with the waypoint, when its distance and angle is less than a threshold
                passed_wps = close_wps[(close_wps['distance_to_line']<max_distance) & (close_wps['angle']<max_angle)]
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
        
        # Construct a GeoDataFrame from graph edges
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

        # Add edge lengths as edge features
        for i in range(0, len(waypoint_connections)):
            orig = waypoint_connections['from'].iloc[i]
            dest = waypoint_connections['to'].iloc[i]
            G[orig][dest]['length'] = waypoint_connections['length'].iloc[i]
       
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