import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import numpy as np
from shapely.geometry import LineString
import io
import cv2
import sys

# add paths for modules
sys.path.append('../models')
sys.path.append('../features')
# import modules
import geometry_utils

def get_bounding_box(gdf):
    """
    Function to return a rectangular bounding box for a set of coordinates or trajectories
    
    :param gdf: GeoDataFrame with geolocations
    :return bbox: GeoDataFrame of a rectangular bounding box
    """
    # get the corners of the bounding box
    bounds = gdf.total_bounds
    # create a GeoDataFrame with the corners as coordinates
    bbox = pd.DataFrame(
        {
            'Lat': [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]],
            'Lon': [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]],
            'ID': [0, 0, 0, 0, 0]
        }
    )
    bbox = gpd.GeoDataFrame(bbox, geometry=gpd.points_from_xy(bbox.Lon, bbox.Lat), crs="EPSG:4326")
    # create a rectangle defined by the corners
    bbox = bbox.groupby(['ID'])['geometry'].apply(lambda x: LineString(x.tolist()))
    return bbox

def traffic_raster_overlay(df, map):
    '''
    creates a hexbin plot as raster overlay from a dataframe
    returns a folium map object
    '''
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10, 10)
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    fig.add_axes(ax)

    # make hexbin plot
    plot = plt.hexbin(df.lon, df.lat, gridsize=6000, mincnt=2, cmap='flag')
    
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.close()
    
    # define a function which returns an image as numpy array from figure
    def get_img_from_fig(fig, dpi=500):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        return img
    
    # get a high-resolution image as numpy array
    plot_img_np = get_img_from_fig(fig)
    
    bounds=[[df.total_bounds[1], df.total_bounds[0]], [df.total_bounds[3], df.total_bounds[2]]]
    folium.raster_layers.ImageOverlay(
        image=plot_img_np,
        name='tracks',
        bounds=bounds,
        opacity=0.2,
        interactive=True,
        mercator_project=True
    ).add_to(map)
    
    return map

def map_accurate_and_simplified_trajectory(accurate, simplified, center=[59, 5], columns=['mmsi', 'geometry'], map=None):
    if map is None:
        map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=8)
    map = accurate.to_traj_gdf()[columns].explore(m=map, color='blue', name='Accurate trajectory', 
                                                  style_kwds={'weight':1, 'color':'black', 'opacity':0.5})
    map = simplified.to_traj_gdf()[columns].explore(m=map, color='red', name='Douglas Peucker simplified trajectory',
                                                    style_kwds={'weight':1, 'color':'blue', 'opacity':0.5})
    return map

def map_prediction_and_ground_truth(predictions, start_node, trajectory, true_path, network, end_node=None, min_passages=3, location='stavanger', opacity=0.2):
    '''
    Plots predictions on an interactive map
    ====================================
    Params:
    predictions: output of self.predict_next_node function
    start_node: Single start node or start node sequence underlying the prediction
    trajectory: original trajectory that we want to make predictions for
    true_path: the actual path in the graph belonging to the original trajectory
    network: the underlying MaritimeTrafficNetwork object
    min_passages: only edges are plotted that have at least min_passages as edge feature
    center: center point of the plotted map
    opacity: opacity of the waypoints and edges of the maritime traffic network
    ====================================
    Returns:
    map: folium map object to display
    '''
    # plot network and basemap
    map = network.map_graph(pruned=True, min_passages=min_passages, 
                            location=location, opacity=opacity)

    # highlight the start node(s)
    wps = network.waypoints[network.waypoints.clusterID.isin(start_node)]
    columns_points = ['clusterID', 'geometry', 'speed', 'cog_before', 'cog_after', 'n_members']  # columns to plot
    columns_hull = ['clusterID', 'convex_hull', 'speed', 'cog_before', 'cog_after', 'n_members']  # columns to plot
    wps.set_geometry('geometry', inplace=True)
    map = wps[columns_points].explore(m=map, name='start nodes', legend=False,
                                      marker_kwds={'radius':3},
                                      style_kwds={'color':'yellow', 'fillColor':'yellow', 'fillOpacity':1, 'opacity':1})
    wps.set_geometry('convex_hull', inplace=True, crs=network.crs)
    map = wps[columns_hull].explore(m=map, name='start nodes convex hulls', legend=False,
                                    style_kwds={'color':'yellow', 'fillColor':'yellow', 'fillOpacity':0.3, 'opacity':1})

    if end_node is not None:
        wps = network.waypoints[network.waypoints.clusterID==end_node]
        wps.set_geometry('geometry', inplace=True)
        map = wps[columns_points].explore(m=map, name='end node', legend=False,
                                          marker_kwds={'radius':3},
                                          style_kwds={'color':'orange', 'fillColor':'orange', 'fillOpacity':1, 'opacity':1})
        wps.set_geometry('convex_hull', inplace=True, crs=network.crs)
        map = wps[columns_hull].explore(m=map, name='end_node convex hull', legend=False,
                                        style_kwds={'color':'orange', 'fillColor':'orange', 'fillOpacity':0.3, 'opacity':1})
        
    
    # generate plottable dataframe of predicted paths
    predicted_paths = pd.DataFrame(columns=['path', 'geometry', 'probability'])
    for key, value in predictions.items():
        path = start_node + [x for x in key]
        line = geometry_utils.node_sequence_to_linestring(path, network.waypoint_connections)
        temp = pd.DataFrame([[tuple(path), line, value]], columns=['path', 'geometry', 'probability'])
        predicted_paths = pd.concat([predicted_paths, temp])
    predicted_paths = gpd.GeoDataFrame(predicted_paths, geometry='geometry', crs=network.crs)
    
    # plot prediction and ground truth
    true_path_line = geometry_utils.get_geo_df(true_path, network.waypoint_connections)
    trajectory = trajectory.to_line_gdf()
    mmsi = trajectory['mmsi'].unique()

    map = trajectory[['mmsi', 'geometry', 'skipsgruppe', 'length', 'bredde']].explore(m=map, style_kwds={'weight':3, 'color':'black', 'opacity':1},
                                                                                      name=f'{mmsi} trajectory')
    map = true_path_line.explore(m=map, style_kwds={'weight':3, 'color':'cyan', 'opacity':1},
                               name=f'{mmsi} closest path')
    for i in range (0, len(predicted_paths)):
        map = predicted_paths.iloc[[i,i]].explore(m=map, style_kwds={'weight':3, 'color':'yellow', 'opacity':1},
                                              name=f'Prediction {i} ({predicted_paths["probability"].iloc[i]*100}% probability)')
    folium.LayerControl().add_to(map)
    return map