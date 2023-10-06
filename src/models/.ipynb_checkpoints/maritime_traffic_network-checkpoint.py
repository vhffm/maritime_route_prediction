import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np
from datetime import timedelta, datetime
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
        self.significant_points = []

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
        self.significant_points = mpd.DouglasPeuckerGeneralizer(self.trajectories).generalize(tolerance=tolerance)
        n_points, n_DP_points = len(self.gdf), len(self.significant_points.to_point_gdf())
        print(f'Number of significant points detected: {n_DP_points} ({n_DP_points/n_points*100:.2f}% of AIS messages)')