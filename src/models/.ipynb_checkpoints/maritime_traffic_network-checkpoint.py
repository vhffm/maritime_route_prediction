import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
from datetime import timedelta, datetime
import time
import folium
import warnings
import sys

class MaritimeTrafficNetwork:
    '''
    DOCSTRING
    '''
    def __init__(self, gdf):
        self.gdf = gdf
        self.trajectories = mpd.TrajectoryCollection(self.gdf, traj_id_col='mmsi', obj_id_col='mmsi')
        self.significant_points_trajectory = []
        self.significant_points = []
        self.waypoints = []

    def get_trajectories_info(self):
        print(f'AIS messages: {len(self.gdf)}')
        print(f'Trajectories: {len(self.trajectories)}')

    def calc_significant_points_DP(self, tolerance):
        '''
        Detect significant turning points with Douglas Peucker algorithm
        :param tolerance: Douglas Peucker algorithm tolerance
        result: self.significant_points is set to a MovingPandas TrajectoryCollection containing
                the significant turning points
        '''
        #
        print(f'Calculating significant turning points with Douglas Peucker algorithm (tolerance = {tolerance}) ...')
        start = time.time()  # start timer
        self.significant_points_trajectory = mpd.DouglasPeuckerGeneralizer(self.trajectories).generalize(tolerance=tolerance)
        n_points, n_DP_points = len(self.gdf), len(self.significant_points_trajectory.to_point_gdf())
        end = time.time()  # end timer
        print(f'Number of significant points detected: {n_DP_points} ({n_DP_points/n_points*100:.2f}% of AIS messages)')
        print(f'Time elapsed: {(end-start)/60:.2f} minutes')

    def calc_waypoints_clustering(self, method='HDBSCAN', min_samples=15, eps=0.008):
        '''
        Compute waypoints by clustering significant turning points
        :param method: Clustering method (supported: DBSCAN and HDBSCAN)
        :param min_points: required parameter for DBSCAN and HDBSCAN
        :param eps: required parameter for DBSCAN
        '''
        start = time.time()  # start timer
        # convert trajectorties to point GeoDataFrame
        significant_points = self.significant_points_trajectory.to_point_gdf()
        # import clustering modules
        from sklearn.cluster import DBSCAN, HDBSCAN
        if method == 'DBSCAN':
            print(f'Calculating waypoints with {method} (eps = {eps}, min_samples = {min_samples}) ...')
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(significant_points[['lat', 'lon']])
        elif method == 'HDBSCAN':
            print(f'Calculating waypoints with {method} (min_samples = {min_samples}) ...')
            clustering = HDBSCAN(min_samples=min_samples).fit(significant_points[['lat', 'lon']])
        else:
            print(f'{method} is not a supported clustering method. Exiting waypoint calculation...')
            return
        
        # compute cluster centroids
        cluster_centroids = pd.DataFrame(columns=['clusterID', 'lat', 'lon', 'convex_hull'])
        for i in range(0, max(clustering.labels_)+1):
            lat = significant_points[clustering.labels_ == i].lat.mean()
            lon = significant_points[clustering.labels_ == i].lon.mean()
            centroid = pd.DataFrame([[i, lat, lon]], columns=['clusterID', 'lat', 'lon'])
            cluster_centroids = pd.concat([cluster_centroids, centroid])
        
        significant_points['clusterID'] = clustering.labels_  # assign clusterID to each waypoint
        
        # convert waypoint and cluster centroid DataFrames to GeoDataFrames
        significant_points = gpd.GeoDataFrame(significant_points, 
                                              geometry=gpd.points_from_xy(significant_points.lon, significant_points.lat), crs="EPSG:4326")
        significant_points.reset_index(inplace=True)
        cluster_centroids = gpd.GeoDataFrame(cluster_centroids, 
                                             geometry=gpd.points_from_xy(cluster_centroids.lon, cluster_centroids.lat), crs="EPSG:4326")
        
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

