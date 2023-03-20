""" Configuration file for VLF_VRF_pred_online_main.py """
""" Code for online VLF """
""" Code version: 2023.01.10 """
""" Previous version: 2019.10.10 (kafka_multiprocess_v04.py) """
""" UPRC - Eva Chondrodima """
""" References: 
P.Tampakis, E.Chondrodima, A.Pikrakis, Y.Theodoridis, K.Pristouris, H.Nakos, E.Petra, T.Dalamagas, A.Kandiros, G.Markakis, I.Maina, S.Kavadas, 
"Sea Area Monitoring and Analysis of Fishing Vessels Activity: The i4sea Big Data Platform," 
2020 21st IEEE International Conference on Mobile Data Management (MDM), Versailles, France, 2020, pp. 275-280, 
doi: 10.1109/MDM48529.2020.00063.
"""

""" Kafka Parameters """

### Folder where Kafka&Zookeeper exist
CFG_KAFKA_FOLDER = './kafka_2.12-2.4.0'

### Set BOOTSTRAP SERVERS
BOOTSTRAP_SERVERS = ["localhost:9092"] # UniPi

### Set Kafka Producer configuration
CFG_KAFKA_P_CONF = {'bootstrap_servers': BOOTSTRAP_SERVERS}

### Set Kafka Consumer configuration
# CFG_KAFKA_C_CONF = {'bootstrap_servers': BOOTSTRAP_SERVERS}
# CFG_KAFKA_C_CONF = {'bootstrap_servers': BOOTSTRAP_SERVERS, 'group_id': 'position-reports-test-1-group'}
CFG_KAFKA_C_CONF = {'bootstrap_servers': BOOTSTRAP_SERVERS, 'auto_offset_reset': 'earliest'}

### Set Number of Consumer - Processes
CFG_PROCESSES_NUM = 4

### Parameter for consumer:'YES': consumer.seek_to_beginning"
CFG_TOPIC_READ_CONSUME_BEGIN = 'NO'
# CFG_TOPIC_READ_CONSUME_BEGIN = 'YES'

### This is the topic input that FLPtool reads (with partition key 'mmsi')
CFG_TOPIC_NAME_IN = 'position-reports-test'

### This is the topic output to which FLPtool writes (with partition key 'mmsi')
CFG_TOPIC_NAME_OUT = 'position-predictions-test'

### File where Kafka Consumer writes
CFG_WRITE_FILE = 'messageskafka.csv'

### Flag for reading from json file (If this is set to 'YES' then we read from json file and not from topic)
CFG_READ_FILE_JSON_FLAG = 'NO' # 'YES' OR 'NO'

### CSV file which Kafka Producer reads
CFG_DATASET = './data/nari_dynamic.csv'



""" PARAMETERS FOR the NN MODEL TRAINED WITH A SUBSET OF BREST DATASET: """

### Trained NN model
CFG_NN_MODEL_PATH = './data/trained_model2.hdf5'
# CFG_NN_MODEL_PATH = './data_model_Brest/model_hdf5/model_Brest.hdf5'

### Trained NN model normalized values
CFG_NN_MODEL_NORM = './data/norm_param_for_trained_model.json'

# ### DateTime format
# CFG_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

### Number of initial records for NN model Prediction
CFG_INIT_POINTS = 11

### NN model Prediction Time Intervals in minutes per predicted point
CFG_DESIRED_DT_PREDICTION_SEC = [300, 900, 1800]
# CFG_DESIRED_DT_PREDICTION_SEC = 900

### Make predictions if the time interval between two consequtive points is greater than CFG_THRESHOLD_TIME_POINTS_SEC(sec)
CFG_THRESHOLD_TIME_POINTS_SEC = 10

### Thresholds for the trained NN model

### Prediction Time Interval in minutes (maximum) If a time interval is higher than CFG_TRAINED_DT_PREDICTION_SEC THEN THE NN CANNOT PREDICT
CFG_TRAINED_DT_PREDICTION_SEC = 1800

### Brest area in which the NN was trained and can make accurate predictions
CFG_THRESHOLD_LON_MAX = 0
CFG_THRESHOLD_LON_MIN = -10
CFG_THRESHOLD_LAT_MAX = 51
CFG_THRESHOLD_LAT_MIN = 45

### Longitude and Latitude Intervals in UTM meters in which the NN was trained and can make accurate predictions
CFG_THRESHOLD_DLON = 100000 #dt=15min
CFG_THRESHOLD_DLAT = 100000 #dt=15min

### Maximum and Minimum Speed in knots (calculated)
CFG_THRESHOLD_SPEED_MAX = 25.7#10.28
CFG_THRESHOLD_SPEED_MIN = 0.5
