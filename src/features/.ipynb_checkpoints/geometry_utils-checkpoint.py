from math import atan2, cos, degrees, pi, radians, sin, sqrt
import shapely
from geopy import distance
from geopy.distance import geodesic
from packaging.version import Version
from shapely.geometry import LineString, Point
import time
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.sparse import coo_matrix


def calculate_initial_compass_bearing(point1, point2):
    """
    Calculate the bearing between two points.

    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `point1: shapely Point
      - `point2: shapely Point
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
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

def aggregate_edges(waypoints, waypoint_connections):
    # refine the graph
    # each edge that intersects the convex hull of another waypoint is divided in segments
    # the segments are added to the adjacency matrix and the original edge is deleted
    print('Aggregating graph edges...')
    start = time.time()  # start timer
    
    n_clusters = len(waypoints)
    coord_dict = {}
    flag = True
    for i in range(0, len(waypoint_connections)):
        edge = waypoint_connections['geometry'].iloc[i]
        mask = edge.intersects(waypoints['convex_hull'])
        subset = waypoints[mask][['clusterID', 'cog_before', 'cog_after', 'geometry']]
        # drop waypoints with traffic direction that does not match edge direction
        subset = subset[np.abs((subset.cog_before + subset.cog_after)/2 - waypoint_connections.direction.iloc[i]) < 30]
        # When we find intersections that match the direction of the edge, we split the edge
        if len(subset)>2:
            # sort by distance
            start_point = edge.boundary.geoms[0]
            subset['distance'] = start_point.distance(subset['geometry'])
            subset.sort_values(by='distance', inplace=True)
            # split line in two segments
            # first segment: between start point and next closest point
            row = subset.clusterID.iloc[0]
            col = subset.clusterID.iloc[1]
            if (row, col) in coord_dict:
                coord_dict[(row, col)] += waypoint_connections['passages'].iloc[i]  # increase the edge weight for each passage
            else:
                coord_dict[(row, col)] = waypoint_connections['passages'].iloc[i]  # create edge if it does not exist yet
            # second segment: between next clostest point and endpoint
            row = subset.clusterID.iloc[1]
            col = waypoint_connections['to'].iloc[i]
            if (row, col) in coord_dict:
                coord_dict[(row, col)] += waypoint_connections['passages'].iloc[i]  # increase the edge weight for each passage
            else:
                coord_dict[(row, col)] = waypoint_connections['passages'].iloc[i]  # create edge if it does not exist yet
        # When we don't find intersections, we keep the original edge
        else:
            row = waypoint_connections['from'].iloc[i]
            col = waypoint_connections['to'].iloc[i]
            if (row, col) in coord_dict:
                coord_dict[(row, col)] += waypoint_connections['passages'].iloc[i]  # increase the edge weight for each passage
            else:
                coord_dict[(row, col)] = waypoint_connections['passages'].iloc[i]  # create edge if it does not exist yet
    
    # create refined adjacency matrix
    row_indices, col_indices = zip(*coord_dict.keys())
    values = list(coord_dict.values())
    A_refined = coo_matrix((values, (row_indices, col_indices)), shape=(n_clusters, n_clusters))
    
    waypoints.set_geometry('geometry', inplace=True)
    waypoint_connections_refined = pd.DataFrame(columns=['from', 'to', 'geometry', 'direction', 'passages'])
    for orig, dest, weight in zip(A_refined.row, A_refined.col, A_refined.data):
        # add linestring as edge
        p1 = waypoints[waypoints.clusterID == orig].geometry
        p2 = waypoints[waypoints.clusterID == dest].geometry
        edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
        # compute the orientation fo the edge (COG)
        p1 = Point(waypoints[waypoints.clusterID == orig].lon, waypoints[waypoints.clusterID == orig].lat)
        p2 = Point(waypoints[waypoints.clusterID == dest].lon, waypoints[waypoints.clusterID == dest].lat)
        direction = calculate_initial_compass_bearing(p1, p2)
        line = pd.DataFrame([[orig, dest, edge, direction, weight]], 
                            columns=['from', 'to', 'geometry', 'direction', 'passages'])
        waypoint_connections_refined = pd.concat([waypoint_connections_refined, line])
    # save result
    waypoint_connections_refined = gpd.GeoDataFrame(waypoint_connections_refined, geometry='geometry', crs=waypoints.crs)
    
    end = time.time()  # end timer
    print(f'Aggregated {len(waypoint_connections)} edges to {len(waypoint_connections_refined)} edges (Time elapsed: {(end-start)/60:.2f} minutes)')
    if len(waypoint_connections) == len(waypoint_connections_refined):
        flag = False
        print(f'Edge aggregation finished.')
    
    return A_refined, waypoint_connections_refined, flag