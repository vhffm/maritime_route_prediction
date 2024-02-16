# maritime_route_prediction
Master Thesis: Graph Analytics for Maritime Route Prediction

This is the code repository for the Master's thesis with the title 'Graph Analytics for Maritime Route Prediction'.
The aim of this project is to create a Maritime Traffic Network (MTN) from AIS data to model maritime traffic and to facilitate route prediction.
This repository does not come with any raw data. 
The raw AIS  data can be downloaded from https://ais-public.kystverket.no/ais-download/.
Additional ship metadata can be downloaded from https://data.kystverket.no/dataset/aarlige-seilas

How to use this library:
- All notebooks except 'RP_Gretel' run in the python environment 'env_geo'. 'RP_Gretel' needs to run in the environment 'env_pyg' 
- Download AIS data and ship metadata (at least 2 months worth of data for a certain geographical region is recommended)
- Generate a maritime traffic network (MTN):
  - Run the notebook 'DATA_preprocess_AIS_data' to clean and prepare the raw AIS data for modelling
  - Run the notebook 'MTN_create_network' to generate a maritime traffic network (MTN)
  - Run the notebook 'EVAL_MTN' to evaluate the network
  - Alternatively, generate and evaluate multiple networks over a set of hyperparameters with the notebook 'EVAL_MTN_hyperparameter_gridsearch'
  - For network refinement, map a set of trajectories to paths on the MTN graph with the notebook 'DATA_map_trajectories_to_paths' and then run 'MTN_refine_network'
  - Visualize networks with the notebook 'VIS_plot_MTN'
- Make route predictions on the network:
  - Map a set of trajectories to paths on the MTN graph with the notebook 'DATA_map_trajectories_to_paths'
  - Train and evaluate route prediction models by running one of the following notebooks
    - 'RP_Dijkstra' (route prediction with destination information)
    - 'RP_Markov' (route prediction without destination information)
    - 'RP_MOGen' (route prediction with or without destination information)
    - 'RP_Gretel' (route prediction with or without destination information). For the Gretel model, input data needs to be pre-processed with the notebook 'DATA_preprocess_for_GRETEL'. The notebook 'EVAL_Gretel' is for evaluation of predictions made with the GRETEL model
  - Alternatively, you can perform a grid search over model hyperparameters with the notebook 'EVAL_RP_hyperparameter_gridsearch'

