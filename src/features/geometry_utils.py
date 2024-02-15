'''
Utility functions for processing geo data
'''

from math import atan2, cos, degrees, pi, radians, sin, sqrt
import shapely
import movingpandas as mpd
import networkx as nx
from geopy import distance
from geopy.distance import geodesic
from packaging.version import Version
from shapely.geometry import LineString, Point, MultiLineString
import time
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.sparse import coo_matrix
from collections import OrderedDict
from shapely import ops


def calculate_initial_compass_bearing(point1, point2):
    '''
    Calculate the bearing between two points using the following formula
    θ = atan2(sin(Δlong).cos(lat2),
              cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    ====================================
    Params:
    point1: shapely Point (in lon, lat format)
    point2: shapely Point (in lon, lat format)
    ====================================
    Returns:
    compass_bearing: (float) the bearing in degrees
    '''
    lat1 = radians(point1.y)
    lat2 = radians(point2.y)
    delta_lon = radians(point2.x - point1.x)
    x = sin(delta_lon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(delta_lon))
    initial_bearing = atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def compass_mean(bearings1, bearings2):
    '''
    Calculate the mean of two compass bearings
    ====================================
    Params:
    bearings1: (float) compass bearing in degrees
    bearings2: (float) compass bearing in degrees
    ====================================
    Returns:
    mean_bearing: (float) the mean bearing in degrees
    '''
    # Convert compass bearings to radians
    rad_bearings1 = np.radians(bearings1)
    rad_bearings2 = np.radians(bearings2)

    # Convert bearings to complex numbers on the unit circle
    complex_bearings1 = np.exp(1j * rad_bearings1)
    complex_bearings2 = np.exp(1j * rad_bearings2)

    # Calculate the mean complex bearing
    mean_complex_bearing = (complex_bearings1 + complex_bearings2) / 2

    # Calculate the mean bearing in degrees
    mean_bearing = np.degrees(np.angle(mean_complex_bearing)) % 360

    return mean_bearing

def mean_bearing(compass_bearings):
    '''
    Compute the mean of a list of compass bearings in degrees
    ====================================
    Params:
    compass_bearings: (list of floats) compass bearings in degrees
    ====================================
    Returns:
    mean_bearing: (float) the mean bearing in degrees
    '''
    # Assuming compass_bearings is a numpy array of compass bearings in degrees
    compass_bearings_rad = np.radians(compass_bearings)  # Convert to radians
    
    # Convert to unit vectors
    unit_vectors = np.column_stack((np.cos(compass_bearings_rad), np.sin(compass_bearings_rad)))
    
    # Calculate the mean vector
    mean_vector = np.mean(unit_vectors, axis=0)
    
    # Convert mean vector back to compass bearing
    mean_bearing = (np.degrees(np.arctan2(mean_vector[1], mean_vector[0])) + 360) % 360
    return mean_bearing

def find_orig_WP(points, waypoints):
    '''
    Given a trajectory and a set of waypoints, find the closest waypoint to the start of the trajectory
    ====================================
    Params:
    points: (GeoDataframe) pointwise representation of a trajectory
    waypoints: (GeoDataframe) waypoints of the maritime traffic network
    ====================================
    Returns:
    orig_WP: (int) ID of the closest waypoint
    idx_orig: (int) index of the trajectory point closest to orig_WP
    dist_orig: (float) distance between trajectory and closest waypoint
    '''
    orig = points.iloc[0].geometry  # get trajectory start point
    # find out if trajectory starts in a stop point
    try:
        orig_speed = points.iloc[0:5].speed.mean()
        if orig_speed < 2:
            mask = (waypoints.speed < 2)
        else:
            orig_cog = calculate_initial_compass_bearing(Point(points.iloc[0].lon, points.iloc[0].lat), 
                                                         Point(points.iloc[9].lon, points.iloc[9].lat)) # get initial cog
            angle = (orig_cog - waypoints.cog_after + 180) % 360 - 180
            mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
    except:
        orig_speed = points.iloc[0:2].speed.mean()
        orig_cog = calculate_initial_compass_bearing(Point(points.iloc[0].lon, points.iloc[0].lat), 
                                                    Point(points.iloc[1].lon, points.iloc[1].lat)) # get initial cog
        angle = (orig_cog - waypoints.cog_after + 180) % 360 - 180
        mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
    distances = orig.distance(waypoints[mask].geometry)
    masked_idx = np.argmin(distances)
    orig_WP = waypoints[mask]['clusterID'].iloc[masked_idx]
    # find trajectory point that is closest to the centroid of the first waypoint
    # this is where we start measuring
    orig_WP_point = waypoints[waypoints.clusterID==orig_WP]['geometry'].item()
    orig_WP_hull = waypoints[waypoints.clusterID==orig_WP]['convex_hull'].item()
    idx_orig = np.argmin(orig_WP_point.distance(points.geometry))
    dist_orig = orig_WP_hull.distance(orig)
    
    return orig_WP, idx_orig, dist_orig

def find_dest_WP(points, waypoints):
    '''
    Given a trajectory and a set of waypoints, find the closest waypoint to the end of the trajectory
    ====================================
    Params:
    points: (GeoDataframe) pointwise representation of a trajectory
    waypoints: (GeoDataframe) waypoints of the maritime traffic network
    ====================================
    Returns:
    dest_WP: (int) ID of the closest waypoint
    idx_dest: (int) index of the trajectory point closest to dest_WP
    dist_dest: (float) distance between trajectory and closest waypoint
    '''
    dest = points.iloc[-1].geometry  # get end point
    try:
        dest_speed = points.iloc[-5:-1].speed.mean()
        if dest_speed < 2:
            mask = (waypoints.speed < 2)
        else:
            dest_cog = calculate_initial_compass_bearing(Point(points.iloc[-10].lon, points.iloc[-10].lat), 
                                                         Point(points.iloc[-1].lon, points.iloc[-1].lat)) # get initial cog
            angle = (dest_cog - waypoints.cog_before + 180) % 360 - 180
            mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
    except:
        dest_speed = points.iloc[-3:-1].speed.mean()
        dest_cog = calculate_initial_compass_bearing(Point(points.iloc[-2].lon, points.iloc[-2].lat), 
                                                     Point(points.iloc[-1].lon, points.iloc[-1].lat)) # get initial cog
        angle = (dest_cog - waypoints.cog_before + 180) % 360 - 180
        mask = np.abs(angle) < 45  # only consider waypoints that have similar direction
    distances = dest.distance(waypoints[mask].geometry)
    masked_idx = np.argmin(distances)
    dest_WP = waypoints[mask]['clusterID'].iloc[masked_idx]
    # find trajectory point that is closest to the centroid of the last waypoint
    # this is where we end measuring
    dest_WP_point = waypoints[waypoints.clusterID==dest_WP]['geometry'].item()
    dest_WP_hull = waypoints[waypoints.clusterID==dest_WP]['convex_hull'].item()
    idx_dest = np.argmin(dest_WP_point.distance(points.geometry))
    dist_dest = dest_WP_hull.distance(dest)
    
    return dest_WP, idx_dest, dist_dest

def find_WP_intersections(points, trajectory, waypoints, G, channel_width):
    '''
    Given a trajectory, find all waypoint intersected by the trajectory in consecutive order
    ====================================
    Params:
    points: (GeoDataframe) pointwise representation of a trajectory
    trajectory: (MovingPandas Trajectory object) a trajectory 
    waypoints: (GeoDataframe) waypoints of the maritime traffic network
    G: (networkx graph) the maritime traffic network graph
    channel_width: (float) the width of the channel around the trajectory, to reduce the graph size for computational efficiency
    ====================================
    Returns:
    cleaned_passages: (list of int) IDs of passed waypoints in consecutive order
    G_channel: (networkx graph) Subgraph of G that lies in a channel of width channel_width around the trajectory
    '''
    # criteria for intersection. Values have been determined empirically
    max_distance = 10
    max_angle = 30
    
    # simplify trajectory
    simplified_trajectory = mpd.DouglasPeuckerGeneralizer(trajectory).generalize(tolerance=10)
    simplified_trajectory.add_direction()
    trajectory_segments = simplified_trajectory.to_line_gdf()
    
    # filter waypoints: only consider waypoints within a certain distance to the trajectory
    distances = trajectory.distance(waypoints['convex_hull'])
    mask = distances <= max_distance
    close_wps = waypoints[mask]
    passages = []  # initialize ordered list of waypoint passages per line segment
    for i in range(0, len(trajectory_segments)-1):
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
        #print('WP COG before: ', WP_cog_before, 'WP COG after: ', WP_cog_after, 'Trajectory COG: ', trajectory_cog)
        close_wps['angle_before'] = np.abs(WP_cog_before - trajectory_cog + 180) % 360 - 180
        close_wps['angle_after'] = np.abs(WP_cog_after - trajectory_cog + 180) % 360 - 180
        # the line segment is associated with the waypoint, when its distance and angle is less than a threshold
        mask = ((close_wps['distance_to_line']<max_distance) & 
                (np.abs(close_wps['angle_before'])<max_angle) & 
                (np.abs(close_wps['angle_after'])<max_angle))
        passed_wps = close_wps[mask]
        #print(close_wps[['clusterID', 'distance_to_line', 'angle_before', 'angle_after']])
        # ensure correct ordering of waypoint passages
        passed_wps.sort_values(by='distance_to_origin', inplace=True)
        passages.extend(passed_wps['clusterID'].tolist())
        passages = list(OrderedDict.fromkeys(passages))

    # get all waypoints in a channel with channel_widt around the trajectory
    mask = distances <= channel_width
    channel_nodes = waypoints[mask]['clusterID'].tolist()
    G_channel = G.subgraph(channel_nodes)
    
    # ensure that all passed lie ahead of each other
    for i in range(0, len(passages)-1):
        # check if we are going backwards
        WP1 = waypoints[waypoints.clusterID==passages[i]]['geometry'].item()  # coordinates of waypoint at beginning of edge sequence
        WP2 = waypoints[waypoints.clusterID==passages[i+1]]['geometry'].item()  # coordinates of waypoint at end of edge sequence
        idx1 = np.argmin(WP1.distance(points['geometry']))  # index of trajectory point closest to beginning of edge sequence
        idx2 = np.argmin(WP2.distance(points['geometry']))  # index of trajectory point closest to end of edge sequence
        # print(passages[i], passages[i+1], idx1, idx2)
        # if we are not going forward, skip next waypoint
        if idx2 <= idx1:
            passages[i+1] = passages[i]
    cleaned_passages = list(OrderedDict.fromkeys(passages))

    # ensure that all passed waypoints are connected somehow
    for i in range(0, len(cleaned_passages)-1):
        has_path = nx.has_path(G_channel, cleaned_passages[i], cleaned_passages[i+1])
        if has_path == False:
            skip_next, skip_this = False, False
            if i <= len(cleaned_passages)-3:
                skip_next = nx.has_path(G_channel, cleaned_passages[i], cleaned_passages[i+2])
            if i >= 1:
                skip_this = nx.has_path(G_channel, cleaned_passages[i-1], cleaned_passages[i+1])
            if skip_next==True:
                cleaned_passages[i+1] = cleaned_passages[i]
                continue
            if skip_this==True:
                cleaned_passages[i] = cleaned_passages[i-1]
    cleaned_passages = list(OrderedDict.fromkeys(cleaned_passages))
    
    return cleaned_passages, G_channel
    

def evaluate_edge_sequence(edge_sequence, connections, idx1, idx2, num_points, eval_traj, eval_points):
    '''
    Compute the SSPD between an edge sequence and a trajectory
    ====================================
    Params:
    edge_sequence: (list of int) list of waypoint IDs
    connections: (GeoDataframe) geometric representation of the edges of the maritime traffic network
    idx1: (int) start index of the trajectory (first point)
    idx2: (int) end index of the trajectory (last point)
    num_points: (int) number of trajectory points (for interpolation purposes)
    eval_traj: (shapely linestring) trajectory to evaluate against
    eval_points: (GeoDataframe) pointwise representation of the trajectory
    waypoints: (GeoDataframe) waypoints of the maritime traffic network
    ====================================
    Returns:
    SSPD: (float) the SSPD between edge sequence and trajectory
    '''
    # create a geometric object (shapely linestring) from the edge sequence
    line = node_sequence_to_linestring(edge_sequence, connections)
    
    # measure distance between the edge sequence and the trajectory
    if idx2 == idx1:
        SSPD = eval_points.distance(line)
    else:
        # get the SSPD between trajectory and edge sequence
        interpolated_points = interpolate_line_to_gdf(line, connections.crs, interval=100)
        SSPD, d12, d21 = sspd(eval_traj, eval_points['geometry'], line, interpolated_points['geometry'])
    return SSPD
    

def sspd(trajectory1, points1, trajectory2, points2):
    '''
    Compute Symmetrized Segment Path Distance between two linestrings
    ====================================
    Params:
    trajectory1: (Shapely linestring) line 1
    points1: (Shapely points) interpolated points on line 1
    trajectory2: (Shapely linestring) line 2
    points2: (Shapely points) interpolated points on line 2
    ====================================
    Returns:
    SSPD: (float) the SSPD between the two lines
    d12: (list of float) distances between points on line 1 and line 2
    d21: (list of float) distances between points on line 2 and line 1
    '''
    d12 = points1.distance(trajectory2)
    SPD12 = np.mean(d12)
    d21 = points2.distance(trajectory1)
    SPD21 = np.mean(d21)
    SSPD = (SPD12 + SPD21) / 2

    return SSPD, d12, d21

def signed_distance_to_line(line, point):
    '''
    Computes the signed distance between a point and a line.
    If the point is right of the line, the distance will be negative
    ====================================
    Params:
    line: (Shapely linestring) a line
    point: (Shapely point) a point
    ====================================
    Returns:
    (float) signed distance
    '''
    # Check if the input is a LineString and a Point
    if not isinstance(line, LineString):
        raise ValueError("The 'line' parameter should be a LineString object.")
    if not isinstance(point, Point):
        raise ValueError("The 'point' parameter should be a Point object.")

    # Calculate vectors AB and AP
    AB = np.array(line.coords[-1]) - np.array(line.coords[0])
    AP = np.array(point.coords) - np.array(line.coords[0])

    # Calculate the cross product
    cross_product = np.cross(AB, AP)

    # Determine the sign of the cross product
    sign = np.sign(cross_product)[0]
    
    # Calculate the distance from the point to the line
    distance = line.distance(point)

    # Return the signed distance
    return sign * distance

def get_geo_df(path, connections):
    '''
    Converts a path represented as a sequence of node IDs into a GeoDataFrame containing the route as a list of LineStrings
    ====================================
    Params:
    path: (list of int) list of waypoint IDs
    connections: (GeoDataframe) geometric representation of the edges of the maritime traffic network
    ====================================
    Returns:
    path_df: (GeoDataframe) the path represented as a GeoDataframe. Each row contains a segment of the path as a linestring
    '''
    path_df = pd.DataFrame(columns=['orig', 'dest', 'geometry'])
    for j in range(0, len(path)-1):
        edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
        temp = pd.DataFrame([[path[j], path[j+1], edge]], columns=['orig', 'dest', 'geometry'])
        path_df = pd.concat([path_df, temp])
    path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=connections.crs)
    return path_df

def node_sequence_to_linestring(sequence, connections):
    '''
    Converts a sequence of node IDs into a shapely LineString
    ====================================
    Params:
    sequence: (list of int) list of waypoint IDs
    connections: GeoDataFrame containing the connections between waypoints
    ====================================
    Returns:
    line: (shapely linestring) edge sequence between waypoints
    '''
    line = []
    for j in range(0, len(sequence)-1):
        #print(sequence[j], sequence[j+1])
        segment = connections[(connections['from'] == sequence[j]) & (connections['to'] == sequence[j+1])].geometry.item()
        line.append(segment)
    line = MultiLineString(line)
    line = ops.linemerge(line)
    return line

def interpolate_line_to_gdf(line, crs, n_points=-1):
    '''
    Interpolates a shapely linestring and returns a GeoDataFrame with the interpolated points
    ====================================
    Params:
    line: (shapely linestring) line to be interpolated
    crs: (int) Coordinate Reference System for GeoDataFrame
    interval: (int) the distance between interpolated points
    ====================================
    Returns:
    points_gdf: GeoDataFrame containing the interpolated points in crs format
    '''
    if n_points == -1:
        interval = 100
    else:
        interval = int(line.length/n_points)
    if interval == 0: interval=1
    points = [line.interpolate(dist) for dist in range(0, int(line.length)+1, interval)]
    points_coords = [Point(point.x, point.y) for point in points]  # interpolated points on edge sequence
    points_df = pd.DataFrame({'geometry': points_coords})
    points_gdf = gpd.GeoDataFrame(points_df, geometry='geometry', crs=crs)
    
    return points_gdf

def clip_trajectory_between_WPs(trajectory, WP1_id, WP2_id, waypoints):
    '''
    Clips a trajectory to a trajectory segment between two waypoints
    ====================================
    Params:
    trajectory: (MovingPandas trajectory object) the trajectory to clip
    WP1_id: (int) ID of the first waypoint
    WP2_id: (int) ID of the second waypoint
    waypoints: (GeoDataFrame) waypoints of the maritime traffic network
    ====================================
    Returns:
    clipped_line: (Shapely LineString) the clipped trajectory
    clipped_points: (GeoDataframe) the clipped trajectory represented as a collection of points
    '''
    traj_points = trajectory.to_point_gdf()
    # clip trajectory to the segment between origin and destination waypoint
    WP1 = waypoints[waypoints.clusterID==WP1_id]['geometry'].item()  # coordinates of waypoint WP1
    WP2 = waypoints[waypoints.clusterID==WP2_id]['geometry'].item()  # coordinates of waypoint WP2
    idx1 = np.argmin(WP1.distance(traj_points.geometry))  # index of trajectory point closest to WP1
    idx2 = np.argmin(WP2.distance(traj_points.geometry))  # index of trajectory point closest to WP2
    # safeguard against errors caused by roundtrips
    if idx2 <= idx1:
        idx1 = 0
        idx2 = -1
    t1 = traj_points.index[idx1]  # get time at passage of waypoint WP1
    t2 = traj_points.index[idx2]  # get time at passage of waypoint WP2
    clipped_line = trajectory.get_linestring_between(t1, t2)  # clipped trajectory as linestring
    clipped_points = traj_points.iloc[idx1:idx2]  # clipped trajectory as points
    
    return clipped_line, clipped_points

def is_valid_path(G, path):
    '''
    Check if a node sequence is a valid path on a graph
    ====================================
    Params:
    path: (list of int) list of node IDs
    G: (networkx graph) a graph
    ====================================
    Returns True if path is a valid path on G
    '''
    return all([(path[i],path[i+1]) in G.edges() for i in range(len(path)-1)])

def split_paths_to_sequences(df, n):
    '''
    splits a path into sub_paths of length n
    example:
    df =    mmsi   path
            4781   [1, 2, 3, 4, 5]
    n = 3
    
    result= mmsi   path
            4781   [1, 2, 3]
            4781   [2, 3, 4]
            4781   [3, 4, 5]

    ====================================
    Params:
    df: (dataframe) dataframe of paths
    n: (int) length of a subpath
    ====================================
    Returns:
    result: (dataframe) a dataframe of paths
    '''
    def create_rows(row, n=2):
        mmsi, path = row
        return [(mmsi, path[i:i+n]) for i in range(len(path) - n + 1)]
    
    # Create a new DataFrame with consecutive elements
    result = pd.DataFrame(
        [item for _, row in test_data.iterrows() for item in create_rows(row, n)],
        columns=['mmsi', 'path']
    )
    return result