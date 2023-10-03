from shapely.geometry import LineString
import pandas as pd
import geopandas as gpd

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