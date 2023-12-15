import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import numpy as np

def load_trajectories(path_prefix, location, crs, dates):
    for i in range(0, len(dates)):
        date = dates[i]
        # Load trajectories from file
        traj_file = date+'_points_'+location+'_cleaned_meta_full_dualSplit_2'
        filename = path_prefix + traj_file + '.parquet'
        traj_gdf = gpd.read_parquet(filename)
        traj_gdf.to_crs(crs, inplace=True)  # Transformation
        
        if i==0:
            all_traj_gdf = traj_gdf
        else:
            all_traj_gdf = pd.concat([all_traj_gdf, traj_gdf])
            
    trajectories = mpd.TrajectoryCollection(all_traj_gdf, traj_id_col='mmsi', obj_id_col='mmsi')
    return trajectories
    