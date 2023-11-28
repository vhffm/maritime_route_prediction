import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from datetime import timedelta, datetime
import warnings
import sys
import argparse
warnings.filterwarnings('ignore')

print("Geopandas has version {}".format(gpd.__version__))
print("Movingpandas has version {}".format(mpd.__version__))

def ais_to_trajectory(filename, size, start, save_to):
    '''
    This script takes raw AIS data as input, cleans and enriches the data and saves it to a parquet file.
    
    Cleaning steps:
    * Drop duplicates (AIS messages can be recorded multiple times by different stations, e.g. satellite, coastal station etc)
      Only the first registered message at a certain location is retained
    * The data is split into trajectories, where each trajectory receives a unique ID. 
      A trajectory is split into sub-trajectories, when the observation gap between AIS messages exceeds 10min and if the resulting
      sub trajectory is longer than 100m
    * Drop trajectories with 'hops' in the AIS messages (Sometimes the GPS location jumps inexplainably between two consecutive timesteps)
    
    Enrichment with metadata:
    * Ship metadata (width, draught, shiptype, shipgroup, name) is added to the raw AIS data
    '''
    
    # read data from file
    df = pd.read_csv(filename, delimiter=';', decimal='.')
    n_messages = len(df)
    print(f'{n_messages} raw AIS messages loaded from file {filename}')
    
    # enrich with ship metadata
    metadata_filename = '../../data/external/seilas-2022.csv'
    df = add_ship_metadata(metadata_filename, df)
    df['date_time_utc'] = pd.to_datetime(df['date_time_utc'])
    
    # convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    df = []  # free memory

    # drop duplicate AIS data
    before = len(gdf)
    #gdf.drop_duplicates(subset = ['mmsi', 'lat', 'lon'],
    #                keep = 'first', inplace=True)
    # filter for nav_status
    nav_status_filter = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, -99] # ships that are not moored, anchored or aground
    filtered_gdf = gdf[gdf['nav_status'].isin(nav_status_filter)]
    nav_filtered = len(filtered_gdf)
    # filter for duplicates (within 5 minutes)
    duplicates = (filtered_gdf.duplicated(subset=['mmsi', 'lat', 'lon'], keep=False))
    timeframe = timedelta(minutes=5)
    duplicates_gdf = filtered_gdf[duplicates]
    duplicates_gdf_sorted = duplicates_gdf.sort_values(by=['mmsi', 'lon', 'lat'])
    duplicates_gdf_sorted['timeframe'] = duplicates_gdf_sorted.groupby(['mmsi', 'lon', 'lat'])['date_time_utc'].diff()
    to_be_deleted_gdf = duplicates_gdf_sorted[duplicates_gdf_sorted['timeframe'].lt(timeframe)]
    indices_to_be_deleted = to_be_deleted_gdf.index  # Get indices to be deleted
    filtered_gdf = filtered_gdf.drop(indices_to_be_deleted)  # Remove rows by index from filtered_gdf
    
    after = len(filtered_gdf)
    print(f'{before-after} superfluous AIS messages dropped, thereof')
    print(f'   {before-nav_filtered} messages with irrelevant nav_status and')
    print(f'   {nav_filtered-after} duplicate messages.')

    # convert to trajectories
    trajectories = mpd.TrajectoryCollection(filtered_gdf.iloc[start:start+size], traj_id_col='mmsi', 
                                            obj_id_col='mmsi', t='date_time_utc')
    
    # add a trajectory observation gap splitter
    # If no AIS message is received for 10 minutes, split the trajectory. Only keep trajectories longer than 500m.
    obs_split_trajectories = mpd.ObservationGapSplitter(trajectories).split(gap=timedelta(minutes=10), min_length=500)
    print(f'Observation Gap splitter split {len(trajectories)} trajectories into {len(obs_split_trajectories)} sub-trajectories')
    # Split trajectories into sub trajectories when a stop longer than 30 seconds is observed (for example ferries). Only keep trajectories longer than 500m.
    split_trajectories = mpd.StopSplitter(obs_split_trajectories).split(max_diameter=50, min_duration=timedelta(minutes=0.5), min_length=500)
    print(f'Stop splitter split {len(obs_split_trajectories)} trajectories into {len(split_trajectories)} sub-trajectories')

    # drop trajectories with 'hops' due to corrupted AIS data
    # We measure the speed of a vessel between consecutive points. If the speed exceeds a certain threshold we discard the trajectory
    split_trajectories.add_speed()  # calculate speed
    speed_thresh = 500 / 3.6  # speed in m/s
    split_gdf = split_trajectories.to_point_gdf()
    bad_track_ids = split_gdf[split_gdf.speed > speed_thresh]['mmsi'].unique()  # IDs that violate the threshold
    valid_track_ids = list(set(split_gdf.mmsi.unique()) - set(bad_track_ids))  # IDs that satisfy the threshold
    split_trajectories = split_trajectories.filter('mmsi', valid_track_ids)  # retain valid trajectories
    print(f'{len(bad_track_ids)} trajectories were found that exceed the speed limit and dropped from the list of trajectories')

    # report about cleaning
    n_retained = len(split_trajectories.to_point_gdf())
    print(f'Cleaning reduced {n_messages} AIS messages to {n_retained} points ({n_retained/n_messages*100:.2f}%)')

    # save to file    
    final_gdf = split_trajectories.to_point_gdf()
    final_gdf['imo_nr'] = final_gdf['imo_nr'].astype(str)
    final_gdf['length'] = final_gdf['length'].astype(str)
    final_gdf.to_parquet(save_to)

def add_ship_metadata(filename, df, join_on='mmsi'):
    '''
    adds ship metadata from specified file to a dataframe
    '''
    # load ship metadata from file
    df_meta = pd.read_csv(filename, delimiter=';', decimal=',', encoding='ISO-8859-1')
    df_meta.rename(columns={'mmsi_nummer':'mmsi'}, inplace=True)  # rename MMSI column
    df_meta.drop_duplicates(subset='mmsi', inplace=True)  # drop duplicate MMSI's
    # merge dataframes on mmsi
    merge_columns = ['mmsi', 'bredde', 'dypgaaende', 'skipstype', 'skipsgruppe', 'fartoynavn']
    df = df.merge(df_meta[merge_columns], on='mmsi', how='left')
    
    # output report about join
    n_matching = len(pd.Series(list(set(df_meta['mmsi']).intersection(set((df['mmsi']))))))
    print(f'Ship metadata has   {df_meta.mmsi.nunique()} unique MMSIs')
    print(f'AIS raw data has    {df.mmsi.nunique()} unique MMSIs')
    print(f'Overlap:            {n_matching} MMSIs')

    return df


def main():
    print('test')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--size')
    parser.add_argument('--start')
    parser.add_argument('--save_to')
    args = parser.parse_args()
    print(args)

    ais_to_trajectory(args.filename, args.size, args.start, args.save_to)

if __name__ == '__main__':
    main()
        
    