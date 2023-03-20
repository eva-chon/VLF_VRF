""" Configuration file for VLF_VRF_train_main.py """
""" Code for VLF VRF """
""" Code version: 2023.01.10 """
""" UPRC - Eva Chondrodima """
""" References: 
E. Chondrodima, N. Pelekis, A. Pikrakis and Y. Theodoridis, "An Efficient LSTM Neural Network-Based Framework for Vessel Location Forecasting," in IEEE Transactions on Intelligent Transportation Systems, doi: 10.1109/TITS.2023.3247993.
E. Chondrodima, P. Mandalis, N. Pelekis and Y. Theodoridis, "Machine Learning Models for Vessel Route Forecasting: An Experimental Comparison," 2022 23rd IEEE International Conference on Mobile Data Management (MDM), Paphos, Cyprus, 2022, pp. 262-269, doi: 10.1109/MDM55031.2022.00056.
P. Mandalis, E. Chondrodima, Y. Kontoulis, N. Pelekis and Y. Theodoridis, "Towards a Unified Vessel Traffic Flow Forecasting Framework," EDBT/ICDT 2023 Joint Conference, Ioannina, Greece, 2023
"""

########################################################################################################################

### ---------------------------------------------- Dataset parameters ---------------------------------------------- ###
cfg_dataset = 'BREST'
# cfg_dataset = 'MARINECADASTRE'

### csv file path that contains the available data
cfg_file_for_read = { 'dynamic': "data/nari_dynamic.csv",
                      'static': "data/nari_static.csv"}
# cfg_file_for_read = {'dynamic':'data/marinecadastre_zone10_2009_02.csv',
#                      'static': 'data/marinecadastre_zone10_2009_02_Vessel_type.csv'}


### Choose specific sea area bounding box [lon_min, lon_max, lat_min, lat_max]
### These parameters should be included in the online prediction code in order to define the sea area in which the NN can make accurate predictions
CFG_THRESHOLD_LON_MIN = -10
CFG_THRESHOLD_LON_MAX = 0
CFG_THRESHOLD_LAT_MIN = 45
CFG_THRESHOLD_LAT_MAX = 51
cfg_sea_area = [CFG_THRESHOLD_LON_MIN, CFG_THRESHOLD_LON_MAX, CFG_THRESHOLD_LAT_MIN, CFG_THRESHOLD_LAT_MAX]
# cfg_sea_area = [-126, -120, 30, 50] #marinecadastre

### Choose specific duration from the available data from dataset_datetime_start to dataset_datetime_end
cfg_dataset_duration = ['2015-10-01', '2016-04-01'] # All Brest data

### Choose specific ship type from the available list:
### 'ALL': all available vessels (including unknown ship types)
### 'ALL_valid': 20-90
### 'fishing': 30
### 'passenger': 60-69
### 'cargo': 70-79
### 'tanker': 80-89
### Vessel types according to:
### https://help.marinetraffic.com/hc/en-us/articles/205579997-What-is-the-significance-of-the-AIS-Shiptype-number-
cfg_ship_type = 'ALL'


### -------------------------------------------- Preprocessing parameters ------------------------------------------ ###
### Parameters for Vessel Position Cleansing & Spatiotemporal-Aware Processing Mechanism

## Minimum & Maximum time intervals between two consequtive points
cfg_gap_period_min_max = [1, 1920] #second

## Number of forecasted locations per trajectory
# Calculation of cfg_val_look_ahead_points/cfg_gap_period_min_max[1]/60 must result in integer
cfg_look_ahead_points = 1  # VLF: 1 # VRF: >1

## Insignificant trajectory elimination (Drop trajectories with less than cfg_traj_points_min_max[0] points)
## Trajectories are partitioned into sub-trajectories when their length exceed cfg_traj_points_min_max[1])
cfg_traj_points_min_max = [20, 1000]

### Minimum & Maximum speed limits
### cfg_speed_min_max[1]: stationery simplification - remove records indicating immobility (vmin)
### cfg_speed_min_max[0]: noise elimination - remove records with invalid speed (vmax)
cfg_speed_min_max = [0.5, 25.7] # m/s
cfg_stop_iterations = 0 # Stop iterations for cleaning speed recursively
cfg_speed_calculate = True

### Parameters for dropping outliers
cfg_drop_outliers = False # Takes True/False values
cfg_drop_outliers_stop_iter = 0 # 1:1loop, 0:many loops
cfg_drop_outliers_sigma = 3

### Dataset split parameters (per cent training & validation)
# cfg_split_method_mmsi_id = 'ID' # Takes values 'ID' or 'MMSI'
cfg_data_split_tr_va = [0.50, 0.25]


### ------------------------------------------------- NN parameters ------------------------------------------------ ###
cfg_nn_model_hneurons = [350, 150] # number of neurons for Hidden [RNN layer, Dense layer]

cfg_python_seed = 9 # seed for initialization
cfg_nn_model_num_epochs = 100 # number of epochs
cfg_nn_early_stop_patience = 20 # stop patience for early stopping procedure

cfg_burned_points_rnn = 10 # the number of points per trajectory that are needed for initializing the NN
########################################################################################################################