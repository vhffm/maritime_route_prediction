import pandas as pd
from ast import literal_eval
import numpy as np

def load_path_training_data(path_prefix, network_name, train_dates, filter=None, data_version=''):
    for i in range(0, len(train_dates)):
        train_date = train_dates[i]
        filename = network_name+'_'+train_date+'_paths'+data_version+'.csv'
        training_data = pd.read_csv(path_prefix+filename)
        training_data['path'] = training_data['path'].apply(literal_eval)
        training_data = training_data[training_data['message']=='success']
        if i==0:
            all_training_data = training_data
        else:
            all_training_data = pd.concat([all_training_data, training_data])
    print(len(all_training_data))
    # filter paths
    all_training_data['skipsgruppe'].fillna('Unknown', inplace=True)
    if filter:
        all_training_data = all_training_data[all_training_data.skipsgruppe==filter]
    
    # extract paths from the training data
    training_paths = all_training_data['path'].tolist()
    print(len(all_training_data), 'training paths loaded.')
    return training_paths

def load_path_test_data(path_prefix, network_name, test_dates, selection_start, selection_end, selection_step, filter=None, data_version=''):
    for i in range(0, len(test_dates)):
        test_date = test_dates[i]
        filename = network_name+'_'+test_date+'_paths'+data_version+'.csv'
        test_data = pd.read_csv(path_prefix+filename)
        test_data['path'] = test_data['path'].apply(literal_eval)
        test_data = test_data[test_data['message']=='success']
        if i==0:
            all_test_data = test_data
        else:
            all_test_data = pd.concat([all_test_data, test_data])
    print(len(all_test_data))        
    # filter paths
    all_test_data['skipsgruppe'].fillna('Unknown', inplace=True)
    if filter:
        all_test_data = all_test_data[all_test_data.skipsgruppe==filter]
    
    # select only mmsi and  path column
    all_test_data = all_test_data[['mmsi', 'path']]
    # select a fraction of test data
    if selection_end == -1:
        selection_end = len(all_test_data)
    selection = np.arange(selection_start, selection_end, selection_step)
    test_paths = all_test_data.iloc[selection]
    print(len(test_paths), 'test paths loaded.')
    return test_paths

def split_path_data(paths, length):
    '''
    split each path in the list of paths into subpaths of length 'length'
    '''
    def create_rows(row, n=2):
        mmsi, path = row
        return [(mmsi, path[i:i+n]) for i in range(len(path) - n + 1)]
    
    # Create a new DataFrame with consecutive elements
    subpaths = pd.DataFrame(
        [item for _, row in paths.iterrows() for item in create_rows(row, length)],
        columns=['mmsi', 'path']
    )
    return  subpaths

def sample_path_data_random(paths, n, seed):
    '''
    randomly samples n rows from a dataframe paths
    '''
    seed_value = seed
    np.random.seed(seed_value)
    if len(paths) >= n:
        return paths.sample(n=n, random_state=seed_value).copy()
    else:
        return paths.copy()

def sample_path_data(paths, start, end, step):
    '''
    samples from a dataframe paths
    '''
    if end == -1:
        end = len(paths)
    selection = np.arange(start, end, step)
    return paths.iloc[selection]
    