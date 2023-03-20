""" Main code file """
""" Code for online VLF """
""" Code version: 2023.01.10 """
""" UPRC - Eva Chondrodima """
""" References: 
P.Tampakis, E.Chondrodima, A.Pikrakis, Y.Theodoridis, K.Pristouris, H.Nakos, E.Petra, T.Dalamagas, A.Kandiros, G.Markakis, I.Maina, S.Kavadas, 
"Sea Area Monitoring and Analysis of Fishing Vessels Activity: The i4sea Big Data Platform," 
2020 21st IEEE International Conference on Mobile Data Management (MDM), Versailles, France, 2020, pp. 275-280, 
doi: 10.1109/MDM48529.2020.00063.
"""
""" Previous version: 2019.10.10 (kafka_multiprocess_v04.py) / UPRC - Eva Chondrodima """

###########################################################################################################
""" Import python files & libraries """
###########################################################################################################
import VLF_VRF_pred_online_config as cfg  # import configuration parameters
import os
import multiprocessing
import time
from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import csv
import json
import pandas
import pyproj
import tensorflow as tf
import logging
import numpy
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



###########################################################################################################
""" NN loss function """
###########################################################################################################
@tf.keras.utils.register_keras_serializable()
def dist_euclidean(y_true, y_pred):
	return tf.keras.backend.mean(tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(
		y_pred - y_true), axis=-1, keepdims=True) + 1e-16), axis=-1)

###########################################################################################################
""" Update buffer """
###########################################################################################################
def update_buffer(id, t, WGS84lon, WGS84lat, speed, lon, lat, df_buffer, myProj, nn_model, norm_param):
	# df_buffer, df, flag, dict_new_row, df_new_row

	""" Update buffer """

	flag = 0  # flag for requirements for prediction #0:cannot make a prediction
	dict_new_row = {} # new record
	df_new_row_tp = pandas.DataFrame() # new record with prediction
	# df_new_row = []
	df = df_buffer[df_buffer['id'] == id].copy()


	""" 
	Check the requirements for making predicitons
	1. The mmsi must be valid:  https://help.marinetraffic.com/hc/en-us/articles/205579997-What-is-the-significance-of-the-AIS-Shiptype-number-
	2. Longitude and Latitude must be in the bounding box area in which the NN was trained
	"""
	if id >= 201000000 and id <= 775999999:
		if WGS84lon < cfg.CFG_THRESHOLD_LON_MAX and WGS84lon > cfg.CFG_THRESHOLD_LON_MIN and \
		   WGS84lat < cfg.CFG_THRESHOLD_LAT_MAX and WGS84lat > cfg.CFG_THRESHOLD_LAT_MIN: # Check if the record is in the boundaries

			UTMlon, UTMlat = myProj(WGS84lon, WGS84lat)  # Conver WGS84 to UTM

			if not df.shape[0] >= 1:  # if this trip has historical points
				# print('---- NNs cannot predict for vessel: %d ----' % id)
				# print('---- This is the first record for this vessel ----')
				dict_new_row = {'id': id,
								't': t,
								'WGS84lon': WGS84lon,
								'WGS84lat': WGS84lat,
								'UTMlon': UTMlon,
								'UTMlat': UTMlat
								}
				df_buffer = df_buffer.append(dict_new_row, ignore_index=True)  # append row to the dataframe
			else: # if this record is the first point of the trip
				d = df[df.t == df.t.max()].copy()
				dt = t - d['t'].values[0]
				if dt < cfg.CFG_THRESHOLD_TIME_POINTS_SEC:  # if this record is less than 1 sec from the previous <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
					# print('---- Invalid record: dt<1sec ----', 'dt=', dt)
					# print('t=', t, 't_prev=', d['t'].values[0])
					if dt < 1:  # if this record is less than 1 sec
						print('---- Invalid record: dt<1sec ----')
						# print('---- ERROR - dt<0: Records for each vessel must be sorted by timestamp ----', 'dt=', dt)
					# print('t=', t, 't_prev=', d['t'].values[0])
				# raise Exception('---- Records for each vessel must be sorted by timestamp ----')
				else:
					dlon = UTMlon - d['UTMlon'].values[0]
					dlat = UTMlat - d['UTMlat'].values[0]
					if dt > cfg.CFG_TRAINED_DT_PREDICTION_SEC or \
							abs(dlon) > cfg.CFG_THRESHOLD_DLON or \
							abs(dlat) > cfg.CFG_THRESHOLD_DLAT:
						# print('---- NNs cannot predict for vessel: %d ----' % id)
						if dt > cfg.CFG_TRAINED_DT_PREDICTION_SEC:
							"""if dt is bigger than the trained then: 
							the whole previous records of the mmsi must be replaced with the new record"""
							df_buffer = df_buffer.drop(df_buffer[df_buffer['id'] == id].index)
							dict_new_row = {'id': id,
											't': t,
											'WGS84lon': WGS84lon,
											'WGS84lat': WGS84lat,
											'UTMlon': UTMlon,
											'UTMlat': UTMlat
											}
							df_buffer = df_buffer.append(dict_new_row, ignore_index=True)  # append row to the dataframe
							# print('---- Delete all the previous records: dt>%d sec ----' % CFG_TRAINED_DT_PREDICTION_SEC)
						else:
							"""if dlon or dlat are bigger than the thresholds then the record is outlier"""
							aa = 0
							# print('---- This record is an outlier ----')
					else:
						dist_m = numpy.sqrt(dlon ** 2 + dlat ** 2)  # euclidean distance in meters
						myspeed = dist_m / dt  # calculate speed
						if myspeed >= cfg.CFG_THRESHOLD_SPEED_MAX or myspeed <= cfg.CFG_THRESHOLD_SPEED_MIN:
							aa = 0
							# print('---- NNs cannot predict for vessel: %d ----' % id)
							# print('---- Invalid speed record (%f) ----' % (myspeed))
						else:
							dict_new_row = {'id': id,
											't': t,
											'WGS84lon': WGS84lon,
											'WGS84lat': WGS84lat,
											'UTMlon': UTMlon,
											'UTMlat': UTMlat,
											'dt': dt,
											'dlon': dlon,
											'dlat': dlat,
											'dist_m': dist_m,
											'myspeed': myspeed
											}

							if df.shape[0] <= cfg.CFG_INIT_POINTS:
								df_buffer = df_buffer.append(dict_new_row, ignore_index=True)  # append row to the datafram
						# print('---- NNs cannot predict for vessel: %d ----' % id)
						# print('---- There are only %d records for this vessel (%d records are needed) ----' % (
						# df.shape[0], CFG_INIT_POINTS))
							else:
								df = df.append(dict_new_row, ignore_index=True)  # append row to the dataframe
								# df_new_row = fun_keras_tf_prediction_model(df, myProj, nn_model, CFG_SC_X_MEAN, CFG_SC_X_STD, CFG_DESIRED_DT_PREDICTION_SEC)  # predict
								# df_buffer = df_buffer.append(df_new_row, ignore_index=True)  # add new row with predictions to the buffer
								flag = 1
	else:
		print("NO VALID MMSI")

	return df_buffer, df, flag


###########################################################################################################
""" Predict longitude & latitude """
###########################################################################################################
def tf_prediction_model(df, myProj, nn_model, norm_param, CFG_DESIRED_DT_PREDICTION_SEC):
	""" Predict longitude & latitude """
	sc_x_mean, sc_x_std = norm_param['sc_x_mean'].values, norm_param['sc_x_std'].values
	###------------------------------------------------------
	# print("Convert series to supervised problem ...")
	x = df[['dt', 'dlon', 'dlat']].copy()
	x['dt_next'] = x['dt'].shift(-1)
	x['dt_next'].iloc[-1] = CFG_DESIRED_DT_PREDICTION_SEC
	x.dropna(inplace=True)
	###------------------------------------------------------
	# print("Reshape input to be 3D [samples, timesteps, features] ...")
	x_3d = x.values.reshape((1, x.shape[0], 4))
	###------------------------------------------------------
	# print("Normalise data (with_mean_std) ....")
	xsc_3d = ((x.values - sc_x_mean) / sc_x_std).reshape((1, x.shape[0], 4))
	###------------------------------------------------------
	# print("Predict ...")
	nn_model.reset_states()
	yhsc = nn_model.predict(xsc_3d, batch_size=1, use_multiprocessing=True)
	###------------------------------------------------------
	# print("Reshape the predictions & Denormalize ...")
	yh = yhsc[0, -1, :] * sc_x_std[1:3] + sc_x_mean[1:3]
	###------------------------------------------------------
	# print("Transform predictions Ds to actual values ...")
	yh_2d_utm = yh + df[['UTMlon', 'UTMlat']].iloc[-1].values
	df.iloc[-1, df.columns.get_loc('pred_UTMlon')] = yh_2d_utm[0] #df['pred_UTMlon'].iloc[-1] = yh_2d_utm[0]
	df.iloc[-1, df.columns.get_loc('pred_UTMlat')] = yh_2d_utm[1] #df['pred_UTMlat'].iloc[-1] = yh_2d_utm[1]

	###------------------------------------------------------
	#print("Convert coordinates: UTM to WGS84 ...")
	lon2, lat2 = myProj(yh_2d_utm[0], yh_2d_utm[1], inverse=True)
	df.iloc[-1, df.columns.get_loc('pred_WGS84lon')] = lon2 #df['pred_WGS84lon'].iloc[-1] = lon2
	df.iloc[-1, df.columns.get_loc('pred_WGS84lat')] = lat2 #df['pred_WGS84lat'].iloc[-1] = lat2
		###------------------------------------------------------
		# print("Calculate new speed for Activity Prediction ...")
	df['pred_speed'] = (numpy.sqrt(
							(df['pred_UTMlon'].iloc[-1] - df['UTMlon'].iloc[-1]) ** 2 +
							(df['pred_UTMlat'].iloc[-1] - df['UTMlat'].iloc[-1]) ** 2)) / CFG_DESIRED_DT_PREDICTION_SEC


	# """Calculate all predictions (# if we want to calculate error)"""
	# print("Reshape the predictions & Denormalize ....")
	# yh = yhsc[0, :, :] * CFG_SC_X_STD[1:3] + CFG_SC_X_MEAN[1:3]
	# print("Transform predictions Ds to actual values ....")
	# yh_2d_utm = yh + df[['lon', 'lat']].shift(-1).iloc[:-1].values
	# df['pred_lon'].iloc[:-1] = yh_2d_utm[:, 0]
	# df['pred_lat'].iloc[:-1] = yh_2d_utm[:, 1]
	# lon2, lat2 = myProj(yh_2d_utm[:, 0], yh_2d_utm[:, 1], inverse=True)
	# df['pred_WGS84lon'].iloc[:-1] = lon2
	# df['pred_WGS84lat'].iloc[:-1] = lat2

	df['pred_t'] = df['t'] + CFG_DESIRED_DT_PREDICTION_SEC

	df_new_row = pandas.DataFrame(data = df.iloc[[-1]], columns = df.columns)
	#df_new_row = df.iloc[[-1]]


	return df_new_row


###########################################################################################################
""" Start Kafka & Zookeeper """
###########################################################################################################
def StartServer():
	"""Start server"""
	print('Start Kafka & Zookeeper ...')
	os.chdir(cfg.CFG_KAFKA_FOLDER)  # cd to kafka folder
	os.system("bin/zookeeper-server-start.sh config/zookeeper.properties")  # Start ZooKeeper
	os.system("bin/kafka-server-start.sh config/server.properties")  # Start Kafka


###########################################################################################################
""" Manage Kafka Topics """
###########################################################################################################
def KafkaTopicsDeleteCreate():
	time.sleep(10)  # sleep in order to start zookeeper&kafka
	def KafkaTopics(topic_name, n_topic_partitions, CFG_KAFKA_P_CONF):
		def fun_delete_kafka_topic(topic_name):
			"""Delete previous kafka topic"""
			print('Delete previous kafka topic ...')
			client = KafkaAdminClient(**CFG_KAFKA_P_CONF)
			if topic_name in client.list_topics():
				print('Topic %s already exists ... deleting ...' % (topic_name))
				client.delete_topics([topic_name])  # Delete kafka topic
			print("List of Topics: %s" % (client.list_topics()))  # See list of topics

		def fun_create_topic(topic_name, n_topic_partitions):
			"""Create topic"""
			print('Create kafka topic %s ...' % (topic_name))
			client = KafkaAdminClient(**CFG_KAFKA_P_CONF)
			topic_list = []
			print('Create topic with %s partitions and replication_factor=1' % (n_topic_partitions))
			topic_list.append(NewTopic(name=topic_name, num_partitions=n_topic_partitions, replication_factor=1))
			client.create_topics(new_topics=topic_list, validate_only=False)
			print("List of Topics: %s" % (client.list_topics()))  # See list of topics
			print("Topic %s description:" % (topic_name))
			print(client.describe_topics([topic_name]))

		fun_delete_kafka_topic(topic_name)
		fun_create_topic(topic_name, n_topic_partitions)
		print("======================================================================================== OK KafkaTopics")

	KafkaTopics(cfg.CFG_TOPIC_NAME_IN, cfg.CFG_PROCESSES_NUM, cfg.CFG_KAFKA_P_CONF)
	KafkaTopics(cfg.CFG_TOPIC_NAME_OUT, cfg.CFG_PROCESSES_NUM, cfg.CFG_KAFKA_P_CONF)


###########################################################################################################
""" Write csv to topic """
###########################################################################################################
def KProducer():
	"""Start Producer"""
	time.sleep(20) # sleep in order to begin the consumers
	print('Start Kafka Producer ...')
	producer = KafkaProducer(**cfg.CFG_KAFKA_P_CONF)
	r = 0

	with open(cfg.CFG_DATASET) as f:
		# reader = csv.DictReader(f, delimiter=';')
		reader = csv.DictReader(f, delimiter=',')  # read csv
		for row in reader: # read each csv row
			# row['t'] = (datetime.datetime.strptime(row['timestamp'], DATETIME_FORMAT)- datetime.datetime(1970, 1, 1)).total_seconds()
			row['mmsi'] = row['sourcemmsi']
			row['speed'] = row['speedoverground']

			row['mmsi'] = str(row['mmsi'])
			row['t'] = str(row['t'])
			row['lon'] = str(row['lon'])
			row['lat'] = str(row['lat'])
			row['speed'] = float(row['speed'])

			key_id = json.dumps(row['mmsi']).encode('utf-8')
			data_row = json.dumps(row).encode('utf-8')  # convert csv row

			producer.send(cfg.CFG_TOPIC_NAME_IN, key=key_id, value=data_row) # send each csv row to consumer
			r = r + 1
			# # print('Producer message:', r, key_id, data_row)
			# print('================== KProducer_read_csv NEW message:', r, key_id, data_row)
	print('Successfully sent data to kafka topic')


###########################################################################################################
""" Consume & Predict """
###########################################################################################################
def KConsumer(consumer_num, CFG_TOPIC_PARTITIONS):
	###########################################################################################################
	def Create_Load_requirements(CFG_NN_MODEL_PATH, CFG_NN_MODEL_NORM):
		COLUMN_NAMES = ['id', 't', 'WGS84lon', 'WGS84lat', 'UTMlon', 'UTMlat',
						'dt', 'dlon', 'dlat',
						'pred_UTMlon', 'pred_UTMlat',
						'pred_WGS84lon', 'pred_WGS84lat']
		df_buffer = pandas.DataFrame(columns=COLUMN_NAMES)  # create dataframe which keeps all the messages
		myProj = pyproj.Proj(
			"+proj=utm +zone=35S, +ellps=WGS84 +datum=WGS84 +units=m +no_defs")  # convert lon,lat to utm
		with tf.device('/cpu:0'):
			nn_model = tf.keras.models.load_model(CFG_NN_MODEL_PATH,
											  custom_objects={'dist_euclidean': dist_euclidean})  # load nn_model
		norm_param = pandas.read_json(CFG_NN_MODEL_NORM)  # load parameters for nn_model normalization
		return df_buffer, myProj, nn_model, norm_param

	###########################################################################################################
	time.sleep(15) # sleep in order to create the topic

	"""Start Producer"""
	producer = KafkaProducer(**cfg.CFG_KAFKA_P_CONF)

	"""Consumer k reads from the k partition - Assign each k consumer to the k partition """
	consumer = KafkaConsumer(**cfg.CFG_KAFKA_C_CONF)
	consumer.assign([TopicPartition(topic=cfg.CFG_TOPIC_NAME_IN, partition=consumer_num)])
	# tp = TopicPartition(topic=read_from_topic, partition=consumer_num)
	# """Consumer with assignment"""
	# consumer.assign([tp])
	# consumer.seek_to_beginning(tp)
	# """Consumer with poll"""
	# consumer.subscribe([read_from_topic])
	# while not consumer._client.poll(): continue

	""" Create-Load requirements """
	df_buffer, myProj, nn_model, norm_param = Create_Load_requirements(
		cfg.CFG_NN_MODEL_PATH, cfg.CFG_NN_MODEL_NORM)

	""" Read message - Update buffer - Make a prediction """
	consumed_messages = 0
	r = 0
	rr_predictions = 0
	rr = 0
	d_concumer_time_start = time.time()
	startt0 = time.time()
	print("----------------------- START CONSUMER ------------------------ ", consumer_num, startt0)
	for message in consumer:
		consumed_messages = consumed_messages + 1
		r = r + 1
		startt_m = time.time()
		d_concumer_time = time.time() - d_concumer_time_start
		startt = time.time()
		# print ("c%d:t%s:p%d:o%d: key=%s value=%s" % (consumer_num, message.topic, message.partition,message.offset,
		#                                       message.key, message.value))
		print('Consumer', consumer_num, '- consumed message: ', r)
		msg = json.loads(message.value.decode('utf-8'))
		message_timestamp = message.timestamp / 1000

		# # parameters for function update_buffer must be int/float
		id, t, WGS84lon, WGS84lat, speed = msg['mmsi'], msg['t'], msg['lon'], msg['lat'], msg['speed']

		# # convert WGS84 coordinates to UTM
		lon, lat = myProj(WGS84lon, WGS84lat)

		# # Update buffer

		df_buffer, df, flag = update_buffer(int(id), float(t), float(WGS84lon), float(WGS84lat),
																	  speed, lon, lat,
																	  df_buffer, myProj, nn_model, norm_param)

		""" If all requirements are met, then predict"""
		if flag == 1:  # if all requirements are met predict
			rr = rr + 1
			df_new_row_tp = []
			dt_prediction_horizon_all = numpy.array(cfg.CFG_DESIRED_DT_PREDICTION_SEC, ndmin=1)
			for ddt in range(0, dt_prediction_horizon_all.shape[0]):  # des_dt.size #len(des_dt)
				des_dt = dt_prediction_horizon_all[ddt].item()
				# print("Predict for", des_dt, "sec")
				#ttflp = time.time()
				df_new_row = []
				df_new_row = tf_prediction_model(df, myProj, nn_model, norm_param, des_dt)  # predict
				#print('predict', time.time() - ttflp)
				# df_new_row['pred_t'] = int(df_new_row['t'] + cfg.CFG_DESIRED_DT_PREDICTION_SEC)
				df_new_row.reset_index(drop=True, inplace=True)
				df_new_row = df_new_row.copy()

				df_new_row['pred_t'] = df_new_row['pred_t'].astype(int)

				if ddt == 0:
					df_new_row_tp = df_new_row.copy()
				else:
					df_new_row = df_new_row[['pred_t',
											 'pred_UTMlon', 'pred_UTMlat',
											 'pred_WGS84lon', 'pred_WGS84lat',
											 'pred_speed'
											 ]].copy()
					df_new_row = df_new_row.rename(columns=lambda x: ('%s_tp%d' % (x, ddt)))  # rename cols
					df_new_row_tp = df_new_row_tp.assign(**df_new_row.iloc[0])

			# update buffer - add new row with predictions
			df_buffer = df_buffer.append(df_new_row_tp, ignore_index=True)
			dict_new_row = df_new_row.to_dict()

			"""-------------- WRITE RESULTS TO KAFKA TOPIC --------------"""
			if "transport_trail" in msg.keys() and msg["transport_trail"]:
				transport_trail = msg["transport_trail"]
			else:
				transport_trail = []
			dt_prediction_horizon_all = numpy.array(cfg.CFG_DESIRED_DT_PREDICTION_SEC,
													ndmin=1)  # Prediction type: flp==1 / tp>1

			print("PREDICTTIME", time.time() - startt)
			""" Write results to topic """

			if dt_prediction_horizon_all.shape[0] > 1:  # Concat TP results
				ttt0 = numpy.concatenate((df_new_row_tp['pred_t'].values,  numpy.concatenate(df_new_row_tp[['pred_t_tp%d' % (
					ddt) for ddt in range(1, dt_prediction_horizon_all.shape[0])]].values, axis=0)), axis=0)
				lon0 = numpy.concatenate((df_new_row_tp['pred_WGS84lon'].values, numpy.concatenate(df_new_row_tp[['pred_WGS84lon_tp%d' % (
						ddt) for ddt in range(1, dt_prediction_horizon_all.shape[0])]].values, axis=0)), axis=0)
				lat0 = numpy.concatenate((df_new_row_tp['pred_WGS84lat'].values, numpy.concatenate(df_new_row_tp[['pred_WGS84lat_tp%d' % (
					ddt) for ddt in range(1, dt_prediction_horizon_all.shape[0])]].values, axis=0)), axis=0)

				transport_trail_tp = transport_trail.copy()
				transport_trail_tp.append({"topic": cfg.CFG_TOPIC_NAME_IN, "timestamp": message_timestamp,
										   "timestamp_start_prediction_process": startt_m})

				for ddt in range(0, dt_prediction_horizon_all.shape[0]):  # Write TP results to dictionary
					dict_res_tp = {
						"prediction_method": "NN",
						"mmsi": msg['mmsi'],
						"prediction_interval": int(dt_prediction_horizon_all[ddt].item()),
						"latest_position_t": t,
						"latest_position_lon": round(float(msg['lon']), 6),
						"latest_position_lat": round(float(msg['lat']), 6),
						"t": int(ttt0[ddt].astype(int)),
						"lon": round(lon0[ddt], 6),
						"lat": round(lat0[ddt], 6),
						# "speed": round(sp0[ddt] * 1.943844, 3),
						"transport_trail": transport_trail_tp
					}
					# dict_res_tp["transport_trail"].append({"topic":read_from_topic, "timestamp": message_timestamp})

					dict_res_tp["transport_trail"][0]["timestamp_end_prediction_process"] = time.time()

					data_row_tp = json.dumps(dict_res_tp).encode('utf-8')
					key = "NN" + "-" + str(msg['mmsi']) + "-" + str(int(dt_prediction_horizon_all[ddt].item()))
					key = key.encode('utf-8')

					producer.send(cfg.CFG_TOPIC_NAME_OUT, key=key, value=data_row_tp)

					# """Write to csv"""
					# dict_res_tp_csv = dict_res_tp.copy()
					# del dict_res_tp_csv['transport_trail']
					# dict_res_tp_csv.update(
					# 	{'trail_' + str(key): val for key, val in dict_res_tp['transport_trail'][0].items()})
					# dict_res_tp_csv = pandas.DataFrame(dict_res_tp_csv, index=[0])
					#
					# if os.path.exists(cfg.CFG_WRITE_FILE):
					# 	dict_res_tp_csv.to_csv(cfg.CFG_WRITE_FILE, mode='a', index=False, header=False)
					# else:
					# 	dict_res_tp_csv.to_csv(cfg.CFG_WRITE_FILE, index=False)
					print("Prediction trail:", dict_res_tp)

		d_concumer_time_start = time.time()
		str0 = "Consumer_num:%d, mmsi:%d, message:%d, flag:%d, prediction:%d, Time(sec):%f, TotalTime(sec):%f, TimeForConsumer(sec):%f" % (
			consumer_num, int(msg['mmsi']), r, flag, rr, time.time() - startt_m, time.time() - startt0,
			d_concumer_time);
		print(str0)
		print("producer messages", rr, "to topic", cfg.CFG_WRITE_FILE)
	print("Finish", rr)



def main():

	print('Start %d Consumers & 1 Producer with %d partitions' % (cfg.CFG_PROCESSES_NUM, cfg.CFG_PROCESSES_NUM))

	jobs = []

	job = multiprocessing.Process(target=StartServer) # Job: Start Kafka & Zookeeper
	jobs.append(job)

	job = multiprocessing.Process(target=KafkaTopicsDeleteCreate) # Job: Delete previous kafka topic & Create new one
	jobs.append(job)

	for i in range(cfg.CFG_PROCESSES_NUM): # Create different consumer jobs
		job = multiprocessing.Process(target=KConsumer, args=(i, cfg.CFG_PROCESSES_NUM))
		jobs.append(job)

	job = multiprocessing.Process(target=KProducer) # Job: Start Producer
	jobs.append(job)

	for job in jobs: # Start jobs
		job.start()

	for job in jobs:
		job.join()

	print("Done!")


if __name__ == "__main__":

	logging.basicConfig(
		format='%(asctime)s.%(msecs)s:%(name)s:%(thread)d:%(levelname)s:%(process)d:%(message)s',
		level=logging.INFO
	)
	main()