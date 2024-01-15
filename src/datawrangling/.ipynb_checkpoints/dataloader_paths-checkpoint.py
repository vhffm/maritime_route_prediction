import pandas as pd
from ast import literal_eval
import numpy as np

def load_path_training_data(path_prefix, network_name, train_dates, data_version=''):
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
    
    # extract paths from the training data
    training_paths = all_training_data['path'].tolist()
    return training_paths

def load_path_test_data(path_prefix, network_name, test_dates, selection_start, selection_end, selection_step, data_version=''):
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
            
    all_test_data = all_test_data[['mmsi', 'path']]

    if selection_end == -1:
        selection_end = len(all_test_data)
    selection = np.arange(selection_start, selection_end, selection_step)
    test_paths = all_test_data.iloc[selection]
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