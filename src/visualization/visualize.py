import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import numpy as np
from shapely.geometry import LineString
import io
import cv2

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
    plot = plt.hexbin(df.lon, df.lat, gridsize=4000, mincnt=3, cmap='flag')
    
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