""" Main code file """
""" Code for VLF VRF """
""" Code version: 2023.01.10 """
""" UPRC - Eva Chondrodima """
""" References: 
E. Chondrodima, N. Pelekis, A. Pikrakis and Y. Theodoridis, "An Efficient LSTM Neural Network-Based Framework for Vessel Location Forecasting," in IEEE Transactions on Intelligent Transportation Systems, doi: 10.1109/TITS.2023.3247993.
E. Chondrodima, P. Mandalis, N. Pelekis and Y. Theodoridis, "Machine Learning Models for Vessel Route Forecasting: An Experimental Comparison," 2022 23rd IEEE International Conference on Mobile Data Management (MDM), Paphos, Cyprus, 2022, pp. 262-269, doi: 10.1109/MDM55031.2022.00056.
P. Mandalis, E. Chondrodima, Y. Kontoulis, N. Pelekis and Y. Theodoridis, "Towards a Unified Vessel Traffic Flow Forecasting Framework," EDBT/ICDT 2023 Joint Conference, Ioannina, Greece, 2023
"""

#######################################################################################################################
# # Import Libraries
#######################################################################################################################
import warnings

warnings.filterwarnings('ignore')
import os
import sys
import time
import pandas
import numpy
import math
import scipy.interpolate
import pyproj
import matplotlib.pyplot as plt
import random
import tensorflow as tf



########################################################################################################################
# # # # # # # # # # # # # # # # # # # # # # # # # #    Basic  Functions    # # # # # # # # # # # # # # # # # # # # # # #
########################################################################################################################
def function_pyproj(lon, lat, transform_from_utm_to_wg84=False, cfg_dataset='BREST'):
    """
    This function performs cartographic transformations
    :param lon values in 4326 projjection system
    :param lat values in 4326 projjection system
    :param transform_from_utm_to_wg84: "True" for transforming UTM to WGS84, "False" for transforming WGS84 TO UTM
    :return: transformed lon and lat
    """
    if cfg_dataset == 'BREST':
        myProj = pyproj.Proj("+proj=utm +zone=30U, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        # myProj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    if transform_from_utm_to_wg84:
        lon2, lat2 = myProj(lon, lat, inverse=True)
        # lon2, lat2 = myProj.transform(lon, lat)
    else:
        lon2, lat2 = myProj(lon, lat)
        # lon2, lat2 = myProj.transform(lon, lat, direction='INVERSE')
    return lon2, lat2


#######################################################################################################################
# # Class Dataset Load & Preprocessing
#######################################################################################################################
class Dataset():
    def __init__(self, cfg_dataset, cfg_file_for_read, cfg_dataset_duration, cfg_sea_area, cfg_ship_type,
                        cfg_gap_period_min_max, cfg_speed_min_max, cfg_stop_iterations, cfg_speed_calculate, cfg_traj_points_min_max,
                        cfg_drop_outliers, cfg_drop_outliers_sigma, cfg_drop_outliers_stop_iter):
        self.cfg_dataset = cfg_dataset
        self.cfg_file_for_read = cfg_file_for_read
        self.cfg_dataset_duration = cfg_dataset_duration
        self.cfg_sea_area = cfg_sea_area
        self.cfg_ship_type = cfg_ship_type
        self.cfg_gap_period_min_max = cfg_gap_period_min_max
        self.cfg_speed_min_max = cfg_speed_min_max
        self.cfg_stop_iterations = cfg_stop_iterations
        self.cfg_speed_calculate = cfg_speed_calculate
        self.cfg_traj_points_min_max = cfg_traj_points_min_max
        self.cfg_drop_outliers = cfg_drop_outliers
        self.cfg_drop_outliers_sigma = cfg_drop_outliers_sigma
        self.cfg_drop_outliers_stop_iter = cfg_drop_outliers_stop_iter

    def load_data_ais_brest(self):
        """
        This function loads the available AIS data in csv format, selects the specific time duration, sea area and ship type
        and makes the necessary columns in the df.

        :param self.cfg_file_for_read: Path for AIS data file
        :param self.cfg_dataset_duration: in the form [datetime_start, datetime_end], e.g. ['2019-12-01', '2019-12-31']
        :param self.cfg_sea_area: sea area bounding box in the form [lon_min, lon_max, lat_min, lat_max]
        :param self.cfg_ship_type: 'ALL' / 'fishing' / 'passenger' / 'cargo' / 'tanker'
        Vessel types according to:
        https://help.marinetraffic.com/hc/en-us/articles/205579997-What-is-the-significance-of-the-AIS-Shiptype-number-

        :return: a dataframe
        """

        ### Read data from csv
        df = pandas.read_csv(self.cfg_file_for_read['dynamic'], header=0, index_col=None)

        ### Each ship is registered in a country (flag).
        ### The country of each ship is encoded in the first three digits of each MMSI number
        ### which should be in the range 201 to 775: https://www.navcen.uscg.gov/maritime-mobile-service-identity
        df = df[(df['sourcemmsi'] >= 201000000) & (df['sourcemmsi'] <= 775999999)].copy() # remove invalid MMSI
        df.reset_index(drop=True, inplace=True)

        ### Transform datetime timestamp to int timestamp in milliseconds
        df['datetime_timestamp'] = pandas.to_datetime(df['t'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Europe/Paris')

        ### Choose specific duration from the available data from dataset_datetime_start to dataset_datetime_end
        dataset_datetime_start = int(pandas.to_datetime(self.cfg_dataset_duration[0]).tz_localize('Europe/Paris').tz_convert('UTC').timestamp())
        dataset_datetime_end = int(pandas.to_datetime(self.cfg_dataset_duration[1]).tz_localize('Europe/Paris').tz_convert('UTC').timestamp())
        df = df[(df['t'] >= dataset_datetime_start) & (df['t'] <= dataset_datetime_end)].copy()

        ### Choose specific sea area bounding box [lon_min, lon_max, lat_min, lat_max]
        df = df[(df['lon'] > self.cfg_sea_area[0]) & (df['lon'] < self.cfg_sea_area[1]) &
                (df['lat'] > self.cfg_sea_area[2]) & (df['lat'] < self.cfg_sea_area[3])].copy()

        ### Sort data according to mmsi and timestamp and calculate time duration between consecutive points
        df.sort_values(by=['sourcemmsi', 't'], ascending=[True, True], inplace=True)
        df['dt'] = (df['t'].groupby(df['sourcemmsi']).diff()).values  # Calculate time duration between consecutive points
        df.reset_index(drop=True, inplace=True)  # Reset df index

        ### Copy-duplicate columns
        df['mmsi'] = df['sourcemmsi'].copy()

        ### Convert speed knots into m/s
        df['speed'] = df['speedoverground'] * 0.514444

        # The 1st traj id is the mmsi, then it changes in order to have many traj ids per mmsi
        df['id'] = df['mmsi'].copy()

        ### Convert lon & lat to UTM
        df['WGS84lon'], df['WGS84lat'] = df['lon'].values, df['lat'].values
        lon2, lat2 = function_pyproj(df['WGS84lon'].values, df['WGS84lat'].values)
        df.update(pandas.DataFrame({'lon': lon2, 'lat': lat2}))

        ### Print data statistics
        print('Dataset Statistics')
        print('Valid AIS records: %d \nNumber of vessels: %d' % (df.shape[0], df.sourcemmsi.unique().shape[0]))
        print('Sampling rate [min, median, mean, max]: [%0.1f, %0.1f, %0.1f, %0.1f]' % (
                                                df.dt.min(), df.dt.median(), df.dt.mean(), df.dt.max()))

        ### Choose specific ship type
        ### Vessel types according to: https://help.marinetraffic.com/hc/en-us/articles/205579997-What-is-the-significance-of-the-AIS-Shiptype-number-
        if self.cfg_ship_type == 'ALL':
            print("ship type: ALL")
        else:
            if self.cfg_ship_type == 'ALL_valid':
                sh_min, sh_max = 20, 90
            elif self.cfg_ship_type == 'fishing':
                sh_min, sh_max = 30, 30
            elif self.cfg_ship_type == 'passenger':
                sh_min, sh_max = 60, 69
            elif self.cfg_ship_type == 'cargo':
                sh_min, sh_max = 70, 79
            elif self.cfg_ship_type == 'tanker':
                sh_min, sh_max = 80, 89
            else:
                print("No valid ship type. Choose: Ship type [ALL, ALL_valid,fishing, passenger, cargo, tanker]")
                sys.exit(1)
            nari_static = pandas.read_csv(self.cfg_file_for_read['static'], header=0, index_col=None)
            nari_static = nari_static[['sourcemmsi', 'shiptype']].copy()
            nari_static.drop_duplicates(subset=['sourcemmsi', 'shiptype'], keep='first', inplace=True)
            nari_static = nari_static.drop((nari_static[nari_static['shiptype'] < sh_min].index) & (
                nari_static[nari_static['shiptype'] > sh_max].index)).reset_index(drop=True)
            nari_static = nari_static.drop_duplicates(subset=['sourcemmsi']).reset_index(drop=True)
            df = pandas.merge(df, nari_static, on='sourcemmsi')
            df.reset_index(drop=True, inplace=True)

        print('Final - Number of points: %d, Number of vessels: %d' % ( df.shape[0], df.mmsi.unique().shape[0]))

        return df

    def load_data_ais(self):
        try:
            if self.cfg_dataset == 'BREST':
                df = self.load_data_ais_brest()
            elif self.cfg_dataset == 'MARINECADASTRE':
                df = self.load_data_ais_marinecadastre()
            else:
                raise Exception("No valid dataset name. Choose: BREST or MARINECADASTRE")
                sys.exit(1)
        except:
            print("No valid dataset name. Choose: BREST")
            sys.exit(1)
        return df

    def _clean_data_1step(self):

        print('..............\n', 'After loading data.... ', 'Number of points:', self.df.shape[0], '  &  ',
              'Number of mmsi:', self.df.mmsi.unique().shape[0], '\n..............')

        print("Drop duplicated values & records with duration less than %d sec...." % (self.cfg_gap_period_min_max[0]))
        self.df.drop_duplicates(keep='first', inplace=True)
        self.df.drop_duplicates(subset=['id', 't'], inplace=True)  # Check if there are points with the same timestamp
        self.df.reset_index(drop=True, inplace=True)  # Reset indexes
        print('After dropping duplicates. ', 'Number of points:', self.df.shape[0], '  &  ',
              'Number of mmsi:', self.df.mmsi.unique().shape[0])
        self.df = self.df.drop(self.df[self.df.dt <= self.cfg_gap_period_min_max[0]].index)  # Delete all data with duration less than cfg_gap_period_min sec (if needed)
        self.df.reset_index(drop=True, inplace=True)  # Reset indexes
        print('After dropping records with duration less than ', self.cfg_gap_period_min_max[0], ' sec. ',
              'Number of points:', self.df.shape[0], '  &  ', 'Number of mmsi:', self.df.mmsi.unique().shape[0],
              '\n..............')

    def _clean_data_due_to_speed_outliers(self):
        def fun_calculate_dt_dlon_dlat_dist_speed_accel(df, cfg_speed_calculate):
            """
            This function calculates differences between consecutive points
            :param df: dataframe, necessary columns: id, t, lon in UTM, lat in UTM, speed (if MarineTraffic provides the dataset)
            :return: dataframe with new columns that include differences: dt, dlon, dlat, dist_m, speed
            """
            # # Calculate speed of movement(in meters/sec) from one location to another
            df['dt'] = (df['t'].groupby(df['id']).diff()).values  # Diff Time in seconds
            df['dlon'] = (df['lon'].groupby(df['id']).diff()).values  # Diff Longitude in meters
            df['dlat'] = (df['lat'].groupby(df['id']).diff()).values  # Diff Latitude in meters
            df['dist_m'] = numpy.sqrt(df.dlon.values ** 2 + df.dlat.values ** 2)  # euclidean distance in meters
            if cfg_speed_calculate:
                df['speed'] = df['dist_m'] / df['dt']
            # df['dspeed'] = (df['speed'].groupby(df['id']).diff()).values
            return df

        self.df = fun_calculate_dt_dlon_dlat_dist_speed_accel(self.df, self.cfg_speed_calculate)
        self.df.reset_index(drop=True, inplace=True)
        self.df = self.df.drop(self.df[self.df.speed < self.cfg_speed_min_max[0]].index)  # Drop points with speed below cfg_speed_min
        self.df = fun_calculate_dt_dlon_dlat_dist_speed_accel(self.df, self.cfg_speed_calculate)
        self.df.reset_index(drop=True, inplace=True)
        print('Number of points after dropping records less than', self.cfg_speed_min_max[0], ': ', self.df.shape[0])
        self.df = self.df.drop(self.df[self.df.speed > self.cfg_speed_min_max[1]].index)  # Drop points with speed > cfg_speed_max
        self.df = fun_calculate_dt_dlon_dlat_dist_speed_accel(self.df, self.cfg_speed_calculate)
        self.df.reset_index(drop=True, inplace=True)
        print('Number of points after dropping records with speed outside [', self.cfg_speed_min_max[0], ' ,',
              self.cfg_speed_min_max[1], ']: ', self.df.shape[0])

        def clean_speed_recursively(df, cfg_gap_period_max, cfg_speed_min_max, cfg_stop_iterations, cfg_speed_calculate):
            """
            This function recursively drops noisy records based on speed limits
            """
            delete_speed_min, n = 0, 0
            while delete_speed_min == 0:
                old_len = df.shape[0]
                df = df.drop(df[(df.dt <= cfg_gap_period_max) & (df.speed < cfg_speed_min_max[0])].index)
                df = fun_calculate_dt_dlon_dlat_dist_speed_accel(df, cfg_speed_calculate)
                df.reset_index(drop=True, inplace=True)
                new_len = df.shape[0]
                n = n + 1
                if old_len == new_len or cfg_stop_iterations == n:
                    delete_speed_min = 1
            print('Number of points after dropping records with speed less than', cfg_speed_min_max[0], ': ',
                  df.shape[0], 'and iterations: ', n)

            delete_speed_max, n = 0, 0
            while delete_speed_max == 0:
                old_len = df.shape[0]
                df = df.drop(df[(df.dt <= cfg_gap_period_max) & (df.speed > cfg_speed_min_max[1])].index)
                df = fun_calculate_dt_dlon_dlat_dist_speed_accel(df, cfg_speed_calculate)
                df.reset_index(drop=True, inplace=True)
                new_len = df.shape[0]
                n = n + 1
                if old_len == new_len or cfg_stop_iterations == n:
                    delete_speed_max = 1
            print('Number of points after dropping records with speed outside [', cfg_speed_min_max[0], ' ,',
                  cfg_speed_min_max[1], ']: ', df.shape[0], 'and iterations: ', n)

            return df

        self.df = clean_speed_recursively(self.df, self.cfg_gap_period_min_max[1], self.cfg_speed_min_max, self.cfg_stop_iterations, self.cfg_speed_calculate)

    def _clean_data_2step(self):

        print('After dropping records with false speed', 'Number of points :',
              self.df.shape[0], '  &  ', 'Number of mmsi:', self.df.mmsi.unique().shape[0], '\n..............')

        print("Drop vessel trajectories (mmsi) with nonaccepted length ....")

        def fun_delete_ids_with_limited_points(df, cfg_traj_points_min):
            """
            This function drops trajectories with less than cfg_traj_points_min points
            :param df: dataframe, necessary columns: id
            :param cfg_traj_points_min:
            :return: dataframe with meaningful trajectories
            """
            gb = df.groupby('id')['id'].transform('size')  # Calculate the number of points per id
            df.drop(df[gb < cfg_traj_points_min].index, inplace=True)  # Drop ids
            df.reset_index(drop=True, inplace=True)  # Reset indexesv
            return df

        self.df = fun_delete_ids_with_limited_points(self.df, self.cfg_traj_points_min_max[0])
        self.df.reset_index(drop=True, inplace=True)  # Reset indexes
        print('After dropping ids with less than ', self.cfg_traj_points_min_max[0], 'Number of points :',
              self.df.shape[0], '  &  ', 'Number of mmsi:', self.df.mmsi.unique().shape[0], '\n..............')

        print('Final basic dataset ---> Number of points: %d, Number of mmsi: %d, Number of id: %d' % (
            self.df.shape[0], self.df.mmsi.unique().shape[0], self.df.id.unique().shape[0]))

    def clean_data(self):
        """
        This function cleans data from noise

        :param cfg_file_for_read: Path for AIS data file
        :param cfg_dataset_duration: in the form [datetime_start, datetime_end], e.g. ['2019-12-01', '2019-12-31']
        :param cfg_sea_area: sea area bounding box in the form [lon_min, lon_max, lat_min, lat_max]
        :param cfg_ship_type: 'ALL' / 'fishing' / 'passenger' / 'cargo' / 'tanker'
        :param cfg_gap_period_min_max:
        :param cfg_speed_min_max: data minimum speed limit & data maximum speed limit
        :param cfg_traj_points_min: minimum number of points for a valid-accepted trajectory
        :return: cleaned AIS data in the form of df
        """
        # self.df = df
        print("Load data & Basic Process ....")
        self.df = self.load_data_ais()
        self._clean_data_1step()
        self._clean_data_due_to_speed_outliers()
        self._clean_data_2step()
        return self.df

    def fun_remove_outliers(self, df):
        """
        This function removes outliers based on dt, dlon, dlat statistics
        :param df: dataframe with AIS data, necessary columns: id, t, lon, lat, dt, dlon, dlat, dataset_tr1_val2_test3
        :param cfg_drop_outliers_sigma: variance (sigma) at gaussian smoothing
        :param cfg_gap_period_max: maximum time interval between consecutive points in seconds
        :param stop_iterations: flag for stopping iterations for removing outliers
        :return: cleaned df
        """

        cfg_gap_period_max = self.cfg_gap_period_min_max[1]
        boxplot_labels = list(range(0, cfg_gap_period_max + 1, 60))
        df['minutes'] = pandas.cut(df.dt, boxplot_labels, labels=[int(i / 60) for i in boxplot_labels[1:]])
        if 'dataset_tr1_val2_test3' in df:
            df_m_s_tr = df[
                df['dataset_tr1_val2_test3'] != 0].copy()  # this is for not taking into account other than training set
        else:
            df_m_s_tr = df.copy()  # this is for not taking ito account other than training set
        df_m_s = pandas.DataFrame()
        df_m_s['dlon_mean'] = df_m_s_tr['dlon'].groupby(df_m_s_tr['minutes']).mean()
        df_m_s['dlon_std'] = df_m_s_tr['dlon'].groupby(df_m_s_tr['minutes']).std()
        df_m_s['dlat_mean'] = df_m_s_tr['dlat'].groupby(df_m_s_tr['minutes']).mean()
        df_m_s['dlat_std'] = df_m_s_tr['dlat'].groupby(df_m_s_tr['minutes']).std()
        delete3sigma = 0
        n = 0
        while delete3sigma == 0:
            old_len = df.shape[0]
            gb = df.groupby(df['minutes'])
            # df_ = pandas.concat([ df[numpy.isnan(df['dt'])].copy(), df[df['minutes'].isna()].copy() ], ignore_index=True)
            df = df[df['minutes'].isna()].copy()
            for i in gb.groups:
                # print(i)
                if gb.groups[i].shape[0] > 0:
                    g = gb.get_group(i).copy()
                    g.drop(g[g['dlon'] < df_m_s['dlon_mean'].loc[i] - df_m_s['dlon_std'].loc[
                        i] * self.cfg_drop_outliers_sigma].index, inplace=True)
                    g.drop(g[g['dlon'] > df_m_s['dlon_mean'].loc[i] + df_m_s['dlon_std'].loc[
                        i] * self.cfg_drop_outliers_sigma].index, inplace=True)
                    g.drop(g[g['dlat'] < df_m_s['dlat_mean'].loc[i] - df_m_s['dlat_std'].loc[
                        i] * self.cfg_drop_outliers_sigma].index, inplace=True)
                    g.drop(g[g['dlat'] > df_m_s['dlat_mean'].loc[i] + df_m_s['dlat_std'].loc[
                        i] * self.cfg_drop_outliers_sigma].index, inplace=True)
                    df = df.append(g)
            df.sort_values(by=['id', 't'], ascending=[True, True], inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['dt'] = (df['t'].groupby(df['id']).diff()).values  # Diff Time in seconds
            df['dlon'] = (df['lon'].groupby(df['id']).diff()).values  # Diff Longitude in meters
            df['dlat'] = (df['lat'].groupby(df['id']).diff()).values  # Diff Latitude in meters
            df.reset_index(drop=True, inplace=True)
            df['minutes'] = pandas.cut(df.dt, boxplot_labels, labels=[int(i / 60) for i in boxplot_labels[1:]])

            new_len = df.shape[0]
            n = n + 1
            if old_len == new_len or self.stop_iterations == n:
                delete3sigma = 1
        print("New df size", new_len)
        return df


    def load_clean_data(self):
        # df = self.load_data_ais()
        df = self.clean_data()
        if self.cfg_drop_outliers:
            print("Remove outliers ....")
            df = self.fun_remove_outliers(df)
        return df


class DatasetProcessed():
    def __init__(self, df, cfg_python_seed, cfg_data_split_tr_va, cfg_gap_period_min_max, cfg_speed_calculate,
                 cfg_traj_points_min_max, cfg_burned_points_rnn):
        self.df = df
        self.cfg_python_seed = cfg_python_seed
        self.rand_seed = cfg_python_seed
        self.cfg_data_split_tr_va = cfg_data_split_tr_va
        self.cfg_gap_period_min_max = cfg_gap_period_min_max
        self.cfg_speed_calculate = cfg_speed_calculate
        self.cfg_traj_points_min_max = cfg_traj_points_min_max
        self.cfg_burned_points_rnn = cfg_burned_points_rnn

    def _split_trajectories(self):
        self.df.sort_values(by=['mmsi', 't'], ascending=[True, True], inplace=True)

        def fun_split_trajectories(df, cfg_gap_period_max, cfg_traj_points_min_max, cfg_speed_calculate):
            """
            This function splits trajectories into meanigful subtrajectories
            :param df: dataframe, necessary columns: id, t
            :param cfg_gap_period_max: maximum time interval between consecutive points in seconds
            :param cfg_traj_points_min_max: minimum number of points for a valid-accepted trajectory & trajectory's maximum number of points
            :return: dataframe with splitted trajectories (different id from mmsi)
            """

            # Split trajectories if dt > cfg_gap_period_max, cfg_speed_min, cfg_speed_max, cfg_traj_points_min
            start = time.time()
            gb = df['dt'].groupby(df['id'])
            gb = [(gb.get_group(y) > cfg_gap_period_max).apply(lambda x: 1 if x else 0).cumsum() for y in gb.groups]
            # Add id for split into df
            gbdf = pandas.concat(gb)
            df['id_dt'] = gbdf.sort_index()
            # Make new ids based on id & id_dt
            # df['id'] = df['id_dt'] + df['id'] + df['id'].max()
            # Make new ids based on id & id_dt
            df['id'] = df.groupby(['id', 'id_dt']).ngroup()
            df.drop(columns=['id_dt'], inplace=True)
            print('..............\n', 'After "split trajectories if dt > cfg_gap_period_max"', cfg_gap_period_max,
                  'Number of points :', df.shape[0], '  &  ', 'Number of mmsi:', df.mmsi.unique().shape[0], '  &  ',
                  'Number of id:', df.id.unique().shape[0])

            def fun_delete_ids_with_limited_points(df, cfg_traj_points_min):
                """
                This function drops trajectories with less than cfg_traj_points_min points
                :param df: dataframe, necessary columns: id
                :param cfg_traj_points_min:
                :return: dataframe with meaningful trajectories
                """
                gb = df.groupby('id')['id'].transform('size')  # Calculate the number of points per id
                df.drop(df[gb < cfg_traj_points_min].index, inplace=True)  # Drop ids
                df.reset_index(drop=True, inplace=True)  # Reset indexesv
                return df

            def fun_calculate_dt_dlon_dlat_dist_speed_accel(df, cfg_speed_calculate):
                """
                This function calculates differences between consecutive points
                :param df: dataframe, necessary columns: id, t, lon in UTM, lat in UTM, speed (if MarineTraffic provides the dataset)
                :return: dataframe with new columns that include differences: dt, dlon, dlat, dist_m, speed
                """
                # # Calculate speed of movement(in meters/sec) from one location to another
                df['dt'] = (df['t'].groupby(df['id']).diff()).values  # Diff Time in seconds
                df['dlon'] = (df['lon'].groupby(df['id']).diff()).values  # Diff Longitude in meters
                df['dlat'] = (df['lat'].groupby(df['id']).diff()).values  # Diff Latitude in meters
                df['dist_m'] = numpy.sqrt(df.dlon.values ** 2 + df.dlat.values ** 2)  # euclidean distance in meters
                if cfg_speed_calculate:
                    df['speed'] = df['dist_m'] / df['dt']
                # df['dspeed'] = (df['speed'].groupby(df['id']).diff()).values
                return df

            df = fun_delete_ids_with_limited_points(df, cfg_traj_points_min_max[0])
            df = fun_calculate_dt_dlon_dlat_dist_speed_accel(df, cfg_speed_calculate)
            df.reset_index(drop=True, inplace=True)
            df['id'] = df.groupby(['id']).ngroup() + 1  # Number each group
            print('SPLIT: After dropping ids with less than ', cfg_traj_points_min_max[0], 'Number of points :',
                  df.shape[0], '  &  ', 'Number of mmsi:', df.mmsi.unique().shape[0], '  &  ', 'Number of id:',
                  df.id.unique().shape[0], '\n..............')
            print(time.time() - start)

            # Split trajectories if num_points > cfg_traj_points_max
            start = time.time()
            df['num'] = pandas.DataFrame(numpy.ones((df.shape[0], 1)))
            df['dt_traj'] = (df['num'].groupby(df['id']).cumsum()).values
            df_ok = pandas.DataFrame()
            while df['dt_traj'].max() > cfg_traj_points_min_max[1]:
                df_ok = pandas.concat(
                    [df_ok, df[numpy.isnan(df['dt_traj'])], df[df['dt_traj'] <= cfg_traj_points_min_max[1]]],
                    sort=False)
                df = df[df['dt_traj'] > cfg_traj_points_min_max[1]].copy()
                df.drop(df[df.groupby('id')['id'].transform('size') < cfg_traj_points_min_max[0]].index,
                        inplace=True)  # delete id with less than n points
                df['id'] = df.groupby(['id']).ngroup() + 1 + max(df_ok['id'].max(), df['id'].max())  # Make new ids
                df['dt_traj'] = (df['num'].groupby(df['id']).cumsum()).values
            df = pandas.concat([df_ok, df], sort=False)
            df.drop(columns=['dt_traj'], inplace=True)
            df.sort_values(by=['id', 't'], ascending=[True, True], inplace=True)
            print('..............\n', 'After "split trajectories if traj_num_points > cfg_traj_points_max"',
                  cfg_traj_points_min_max[1], 'Number of points :', df.shape[0], '  &  ', 'Number of mmsi:',
                  df.mmsi.unique().shape[0], '  &  ', 'Number of id:', df.id.unique().shape[0])

            df = fun_delete_ids_with_limited_points(df, cfg_traj_points_min_max[0])
            df = fun_calculate_dt_dlon_dlat_dist_speed_accel(df, cfg_speed_calculate)
            df.reset_index(drop=True, inplace=True)
            df['id'] = df.groupby(['id']).ngroup() + 1  # Number each group
            print('SPLIT: After dropping ids with less than ', cfg_traj_points_min_max[0], 'Number of points :',
                  df.shape[0], '  &  ', 'Number of mmsi:', df.mmsi.unique().shape[0], '  &  ', 'Number of id:',
                  df.id.unique().shape[0], '\n..............')
            print(time.time() - start)

            return df

        self.df = fun_split_trajectories(self.df, self.cfg_gap_period_min_max[1], self.cfg_traj_points_min_max, self.cfg_speed_calculate)

    def _fun_rename_traj_ids(self):
        """
        Thif function gives new code names-numbers to trajectories ids
        :param df: dataframe, necessary columns: id, t
        :return: trajectories with new code names-numbers in form of dataframe
        """
        self.df = self.df.sort_values(by=['t'], ascending=[True], ignore_index=True).copy()
        df_ = self.df.drop_duplicates(subset=['id'], keep='first').copy()
        df_['id_dt'] = df_.groupby(['id']).ngroup() + 1
        df_ = df_[['id_dt', 'id']].copy()
        df_.rename(columns={"id": "d_id"}, inplace=True)
        self.df = pandas.merge(df_, self.df, left_on='d_id', right_on='id')
        self.df.drop(columns=['d_id', 'id'], inplace=True)
        self.df.rename(columns={"id_dt": "id"}, inplace=True)
        self.df.sort_values(by=['id', 't'], ascending=[True, True], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def _fun_split_into_sets(self):
        """
        This function split the available dataset into 3 subsets: training, validation, testing
        :param df: dataframe, necessary columns: id
        :param cfg_data_split_tr_va:
        :return: dataframe with two new columns that indicate the subset: [1]training, [2]validation, [3]testing
        """
        self.ntr = int(self.df['id'].unique().shape[0] * self.cfg_data_split_tr_va[0])
        self.nva = int(self.df['id'].unique().shape[0] * self.cfg_data_split_tr_va[1])
        # nte = df['id'].unique().shape[0] - (ntr+nva)
        ids = self.df['id'].unique()
        ids_tr, ids_va, ids_te = ids[:self.ntr].copy(), ids[self.ntr:self.ntr + self.nva].copy(), ids[self.ntr + self.nva:].copy()
        self.df['dataset_tr1_val2_test3'] = self.df['id'].copy()
        self.df.loc[self.df['id'].isin(ids_tr), 'dataset_tr1_val2_test3'] = 1
        self.df.loc[self.df['id'].isin(ids_va), 'dataset_tr1_val2_test3'] = 2
        self.df.loc[self.df['id'].isin(ids_te), 'dataset_tr1_val2_test3'] = 3
        self.df['id_split'] = self.df['id'].copy()


    def _fun_shuffle_id(self):
        """
        This functions shuffles trajectories per subset (training/validation/testing)
        :param df: dataframe, necessary columns: id, t, dataset_tr1_val2_test3
        :param rand_seed: seed number
        :return: shuffled trajectories with new code names-numbers in form of dataframe
        """

        def fun_shuffle(df, rand_seed):
            """
            This functions shuffles trajectories
            :param df: dataframe, necessary columns: id, t
            :param rand_seed: seed number
            :return: shuffled trajectories with new code names-numbers in form of dataframe
            """
            groups = [df for _, df in df.groupby('id')]
            random.shuffle(groups, random.seed(rand_seed))
            pandas.concat(groups).reset_index(drop=True)
            df = pandas.concat(groups).reset_index(drop=True)
            df['id'] = df['id'].groupby(df['id'], sort=False).ngroup() + 1
            df.sort_values(by=['id', 't'], ascending=[True, True], inplace=True, ignore_index=True)
            return df

        df_tr = fun_shuffle(self.df[self.df['dataset_tr1_val2_test3'] == 1].copy(), self.rand_seed)
        df_tr['id'] = df_tr['id'].groupby(df_tr['id'], sort=False).ngroup() + 1

        df_va = fun_shuffle(self.df[self.df['dataset_tr1_val2_test3'] == 2].copy(), self.rand_seed)
        df_va['id'] = df_va['id'].groupby(df_va['id'], sort=False).ngroup() + 1 + df_tr.id.max()

        df_te = fun_shuffle(self.df[self.df['dataset_tr1_val2_test3'] == 3].copy(), self.rand_seed)
        df_te['id'] = df_te['id'].groupby(df_te['id'], sort=False).ngroup() + 1 + df_va.id.max()

        self.df = pandas.concat([df_tr, df_va, df_te], sort=False, ignore_index=True)
        self.df.sort_values(by=['id', 't'], ascending=[True, True], inplace=True, ignore_index=True)

    def split_trajectories_sets_shuffle(self):
        self._split_trajectories()
        self._fun_rename_traj_ids()
        self._fun_split_into_sets()
        self._fun_shuffle_id()
        return self.df


class DatasetTransformed():
    def __init__(self, df, cfg_python_seed, cfg_data_split_tr_va, cfg_look_ahead_points, cfg_gap_period_min_max,
                 cfg_traj_points_min_max, cfg_burned_points_rnn,
                 path_savename, file_savename):
        self.df = df
        self.cfg_python_seed = cfg_python_seed
        self.rand_seed = cfg_python_seed
        self.cfg_data_split_tr_va = cfg_data_split_tr_va
        self.cfg_look_ahead_points = cfg_look_ahead_points
        self.cfg_gap_period_min_max = cfg_gap_period_min_max
        self.cfg_traj_points_min_max = cfg_traj_points_min_max
        self.cfg_burned_points_rnn = cfg_burned_points_rnn
        self.path_savename = path_savename
        self.file_savename = file_savename
        self.cfg_val_look_ahead_dt = self.cfg_gap_period_min_max[1] / self.cfg_look_ahead_points
        if (self.cfg_gap_period_min_max[1] / 60) % self.cfg_look_ahead_points:
            print("Calculation of cfg_look_ahead_points/cfg_gap_period_min_max[1]/60 must result in integer")
            sys.exit(1)

        ##Set input/output features
        self.cfg_features_init = ['mmsi', 'id', 'dataset_tr1_val2_test3']
        self.cfg_features_all = self.cfg_features_init + ['t', 'lon', 'lat', 'WGS84lon', 'WGS84lat', 'dt', 'dlon', 'dlat', 'dist_m', 'speed']
        self.cfg_features_inputs = ['dt(t)', 'dlon(t)', 'dlat(t)', 'dt(t+1)']  # Create input feature columns
        self.cfg_features_outputs = ['dlon(t+1)', 'dlat(t+1)']  # name of the features we want to predict
        if self.cfg_look_ahead_points > 1:
            self.cfg_features_inputs = ['dt(t)', 'dlon(t)', 'dlat(t)']
            self.cfg_features_outputs = [(x % (y)) for x in ['dlon(t+%d)', 'dlat(t+%d)'] for y in range(
                                                                                1, self.cfg_look_ahead_points + 1)]


    def _dataset_to_supervised(self):
        class TrajectoriesToSupervised:
            """ Class for transforming trajectories to supervised problem & interpolating output """

            def __init__(self, data, cfg_features_init):
                self.data = data
                self.cfg_features_init = cfg_features_init

            def trajectories_to_supervised_input(self, n_in):
                """
                Frame a time series as a supervised learning dataset.
                Arguments:
                    n_in: Number of lag observations as input (X)
                Returns:
                    Pandas DataFrame of series framed for supervised learning
                """
                """--- Make inputs  ---"""
                cols = self.data.columns.values
                cols = cols[~numpy.isin(cols, self.cfg_features_init)]
                data = self.data.copy()
                data['id2'] = data['id'].copy()
                agg = pandas.DataFrame()
                for iii in range(n_in - 1, -1, -1):
                    df0_shifted__1 = data.groupby(['id2']).shift(iii).copy()
                    if iii == 0:
                        df0_shifted__1.rename(columns=lambda x: ('%s(t)' % (x)) if x in cols else x, inplace=True)
                    else:
                        df0_shifted__1.rename(columns=lambda x: ('%s(t-%d)' % (x, iii)) if x in cols else x,
                                              inplace=True)
                    agg = pandas.concat([agg, df0_shifted__1], axis=1)
                agg = agg.loc[:, ~agg.columns.duplicated(keep='last')].copy()
                return agg

            def trajectories_to_supervised_output_flp(self, n_out):
                """
                Frame a time series as a supervised learning dataset.
                Arguments:
                    n_out: Number of observations as output (y).
                Returns:
                    Pandas DataFrame of series framed for supervised learning.
                """
                """--- Make outputs  ---"""
                cols = self.data.columns.values
                cols = cols[~numpy.isin(cols, self.cfg_features_init)]
                data = self.data.copy()
                data['id2'] = data['id'].copy()
                agg = pandas.DataFrame()
                for iii in range(1, n_out + 1):
                    df0_shifted_1 = data.groupby(['id2']).shift(-iii).copy()
                    df0_shifted_1.rename(columns=lambda x: ('%s(t+%d)' % (x, iii)) if x in cols else x,
                                         inplace=True)
                    agg = pandas.concat([agg, df0_shifted_1], axis=1)
                return agg

            def trajectories_to_supervised_input_output_flp(self, n_in, n_out):
                aggi = self.trajectories_to_supervised_input(n_in)
                aggo = self.trajectories_to_supervised_output_flp(n_out)
                agg = pandas.concat([aggi, aggo], axis=1)
                agg = agg.loc[:, ~agg.columns.duplicated(keep='first')].copy()
                return agg

            def trajectories_to_supervised_output_interpolated(self, df0, cfg_val_look_ahead_dt,
                                                               cfg_look_ahead_points, interp_kind='linear'):
                def fun_interpolate_output_fast(df0, cfg_val_look_ahead_dt, cfg_look_ahead_points, interp_kind):
                    # f_id = [mmsi_id for mmsi_id in df0['id'].unique()]
                    t = time.time()
                    f_lon = [scipy.interpolate.interp1d(df0[df0['id'] == mmsi_id]['t(t)'],
                                                        df0[df0['id'] == mmsi_id]['lon(t)'], kind=interp_kind) for
                             mmsi_id in df0['id'].unique()]
                    f_lat = [scipy.interpolate.interp1d(df0[df0['id'] == mmsi_id]['t(t)'],
                                                        df0[df0['id'] == mmsi_id]['lat(t)'], kind=interp_kind) for
                             mmsi_id in df0['id'].unique()]
                    # print(time.time()-t)

                    for la_ind, la in enumerate(range(1, cfg_look_ahead_points + 1)):
                        df0['t(t+%d)' % (la_ind + 1)] = df0['t(t)'] + (la * cfg_val_look_ahead_dt)
                        df0['lon(t+%d)' % (la_ind + 1)], df0['lat(t+%d)' % (la_ind + 1)] = numpy.nan, numpy.nan
                        df0['dt(t+%d)' % (la_ind + 1)] = numpy.nan
                        df0['dlon(t+%d)' % (la_ind + 1)], df0['dlat(t+%d)' % (la_ind + 1)] = numpy.nan, numpy.nan
                        df0['dist_m(t+%d)' % (la_ind + 1)], df0['speed(t+%d)' % (la_ind + 1)] = numpy.nan, numpy.nan

                    for la_ind, la in enumerate(range(1, cfg_look_ahead_points + 1)):
                        df0['max_t'] = df0.groupby('id')['t(t)'].transform('max')
                        df0['max_t'] = df0['max_t'] - df0['t(t+%d)' % (la_ind + 1)]
                        df0.loc[df0[df0['max_t'] < 0].index, 't(t+%d)' % (la_ind + 1)] = numpy.nan
                        df0['dt(t+%d)' % (la_ind + 1)] = df0['t(t+%d)' % (la_ind + 1)] - df0['t(t)']
                    df0.drop(columns=['max_t'], inplace=True)

                    if df0.id.unique().shape[0] != df0.id.max():
                        print(exit)
                        exit()

                    def fun_interpolate_lon_lat(la_ind, df0):
                        t = time.time()
                        df00 = pandas.DataFrame()
                        df00['lon(t+%d)' % (la_ind + 1)] = numpy.hstack(
                            [f_lon[mmsi_id - 1](df0[df0['id'] == mmsi_id]['t(t+%d)' % (la_ind + 1)]) for mmsi_id in
                             df0['id'].unique()])
                        df00['lat(t+%d)' % (la_ind + 1)] = numpy.hstack(
                            [f_lat[mmsi_id - 1](df0[df0['id'] == mmsi_id]['t(t+%d)' % (la_ind + 1)]) for mmsi_id in
                             df0['id'].unique()])
                        # for i in range(df0.id.unique().shape[0]):
                        #     df0.loc[df_index[i], 'lon(t+%d)' % (la_ind + 1)] = df_lon[i]
                        #     df0.loc[df_index[i], 'lat(t+%d)' % (la_ind + 1)] = df_lat[i]
                        # df0['lon(t+%d)' % (la_ind + 1)] = dask_array.hstack(df_lon).compute()
                        # df0['lat(t+%d)' % (la_ind + 1)] = dask_array.hstack(df_lat).compute()
                        df00['dlon(t+%d)' % (la_ind + 1)] = df00['lon(t+%d)' % (la_ind + 1)] - df0['lon(t)']
                        df00['dlat(t+%d)' % (la_ind + 1)] = df00['lat(t+%d)' % (la_ind + 1)] - df0['lat(t)']
                        df00['dist_m(t+%d)' % (la_ind + 1)] = numpy.sqrt(
                            df00['dlon(t+%d)' % (la_ind + 1)].values ** 2 + df00[
                                'dlat(t+%d)' % (la_ind + 1)].values ** 2)  # euclidean distance in meters
                        df00['speed(t+%d)' % (la_ind + 1)] = df00['dist_m(t+%d)' % (la_ind + 1)] / df0[
                            'dt(t+%d)' % (la_ind + 1)]
                        print(la_ind + 1, ": ", time.time() - t)
                        return df00

                    # t = time.time()
                    # manager = multiprocessing.Manager()
                    # return_list = manager.list()
                    # jobs = []
                    # for la_ind in range(0, cfg_look_ahead_points + 1):  # for i in range(CFG_PROCESSES_NUM):  # Create different consumer jobs
                    #     job = multiprocessing.Process(target=fun_interpolate_lon_lat, args=(la_ind, df0, df00, return_list))
                    #     jobs.append(job)
                    # print("---------------------------------------------------------")
                    # for job in jobs: job.start()
                    # for job in jobs: job.join()

                    for la_ind in range(0, cfg_look_ahead_points):
                        t = time.time()
                        df00 = fun_interpolate_lon_lat(la_ind, df0)
                        df0[df00.columns] = df00.copy()
                    print("interpolate total t: ", time.time() - t)
                    # print(df0.columns)

                    return df0

                dfsp_interp = fun_interpolate_output_fast(df0, cfg_val_look_ahead_dt, cfg_look_ahead_points,
                                                          interp_kind)

                for la in range(1, cfg_look_ahead_points + 1):
                    dfsp_interp['WGS84lon(t+%d)' % (la)], dfsp_interp['WGS84lat(t+%d)' % (la)] = function_pyproj(
                        dfsp_interp['lon(t+%d)' % (la)].values, dfsp_interp['lat(t+%d)' % (la)].values, True)
                return dfsp_interp

            def trajectories_to_supervised_input_output_interpolated(self, n_in, cfg_val_look_ahead_dt,
                                                                     cfg_look_ahead_points, interp_kind):
                df0 = self.trajectories_to_supervised_input(n_in)
                agg = self.trajectories_to_supervised_output_interpolated(df0, cfg_val_look_ahead_dt,
                                                                          cfg_look_ahead_points)
                return agg

        self.df = self.df[self.cfg_features_all].copy().dropna().reset_index(drop=True)  # Keep specific columns & drop nan & reset index
        sp = TrajectoriesToSupervised(self.df, self.cfg_features_init)
        if self.cfg_look_ahead_points > 1:
            self.dfsp = sp.trajectories_to_supervised_input_output_interpolated(1, self.cfg_val_look_ahead_dt, self.cfg_look_ahead_points, interp_kind='linear')
            self.dfsp.dropna(inplace=True)
            self.dfsp.reset_index(drop=True, inplace=True)
        else:
            self.dfsp = sp.trajectories_to_supervised_input_output_flp(1, 1)
            self.dfsp.dropna(inplace=True)
            self.dfsp.reset_index(drop=True, inplace=True)


    def _fun_set_burned_points(self):
        """
        This function
        :param df: dataframe, necessary columns: id,
        :param cfg_burned_points_rnn: the number of points per trajectory that are needed for initializing the NN
        :return: df: dataframe with extra columns: burned, cumcount
        """
        self.dfsp['burned'] = 0
        self.dfsp['cumcount'] = self.dfsp.groupby('id').cumcount() + 1
        self.dfsp.loc[self.dfsp[self.dfsp['cumcount'] < self.cfg_burned_points_rnn].index, ['burned']] = 1

    def _fun_set_the_minute_labels(self):
        """
        This function set the minute labels, e.g. 0-59sec:1min, 60-119sec:2min, 120-179sec:3min, etc
        :param dfsp: dataframe, necessary columns: id, dt(t)
        :param self.cfg_look_ahead_points: number of steps in output
        :return: dfsp: dataframe with extra columns: minute labels for input & output, number of points for each id
        """
        dt_per_min = 60  # 60sec
        b_labels = list(range(0, math.ceil(self.dfsp['dt(t)'].max() / dt_per_min) * dt_per_min + 1, dt_per_min))
        self.dfsp['minutes_labels(t)'] = pandas.cut(self.dfsp['dt(t)'], b_labels,
                                                    labels=[int(iii / 60) for iii in b_labels[1:]])
        self.dfsp['minutes_labels(t)'] = self.dfsp['minutes_labels(t)'].astype(int)
        self.dfsp['traj_steps_all'] = self.dfsp.groupby('id').cumcount() + 1

        for la in range(1, self.cfg_look_ahead_points + 1):
            self.dfsp['minutes_labels(t+%d)' % (la)] = pandas.cut(self.dfsp['dt(t+%d)' % (la)], b_labels, labels=[
                int(iii / 60) for iii in b_labels[1:]])
            self.dfsp['minutes_labels(t+%d)' % (la)] = self.dfsp['minutes_labels(t+%d)' % (la)].astype(int)

    def dataset_to_supervised_with_labels(self):
        self._dataset_to_supervised()
        self._fun_set_burned_points()  # Set the burned points (points needed for setting LSTM states)
        self._fun_set_the_minute_labels()  # Set the minute labels for each dt
        return self.dfsp

    def fun_dataset_normalised(self):
        self.norm_param = self.dfsp.loc[self.dfsp['dataset_tr1_val2_test3'] == 1, self.cfg_features_inputs].agg(
            ['mean', 'std', 'min', 'max'])
        self.norm_param.rename(columns=lambda x: str(x)[:str(x).find('(')], inplace=True)
        pandas.DataFrame(
            {'sc_x_mean': self.norm_param.loc['mean'].values,
             'sc_x_std': self.norm_param.loc['std'].values}).to_json(
            "%s/%s_norm_param_mean_std.json" % (
                self.path_savename, self.file_savename))  # Save normalized parameters
        self.norm_param = self.norm_param.loc[:, ~self.norm_param.columns.duplicated()]

        def fun_dfsp_normalised(dfsp, norm_param):
            """
            This functions normalizes specific columns in dfsp with mean and std based on norm_param
            :param dfsp: dataframe
            :param norm_param: dataframe with columns for features for normalization & rows for values mean, std, min, max
            :return: a dataframe with all necessary normalized columns
            """
            dfsp_normalised = pandas.DataFrame()
            for x in norm_param.columns.values:
                cols = dfsp.loc[:, dfsp.columns.str.startswith(x)].columns.values
                for i in cols:
                    dfsp_normalised[('%s' % (i))] = (dfsp[('%s' % (i))] - norm_param.loc['mean', ('%s' % (x))]) / \
                                                    norm_param.loc['std', ('%s' % (x))]
            return dfsp_normalised

        self.dfsp_norm = fun_dfsp_normalised(self.dfsp, self.norm_param)
        return self.dfsp_norm, self.norm_param

    def fun_dataset_padded(self):
        points_all = self.dfsp.groupby('id').cumcount().max() + 1  # calculate maximum trajectory length

        def fun_all_pad_sequences_TP_df(dfsp, dfsp_norm, points_all):
            """
            This function conducts pre-zero padding at the trajectories of dfsp based on the maximum trajectory length
            :param dfsp: dataframe with trajectories
            :param dfsp_norm: dataframe with normalized trajectories
            :param points_all: maximum trajectory length
            :return: dfsp_pad, dfsp_norm_pad: dataframes with original and normalized padded trajectories
            """

            def pad_sequences_TP_df(dfsp, points_all):
                """
                This function conducts pre-zero padding at the trajectories of dfsp based on the maximum trajectory length
                :param dfsp: dataframe with trajectories
                :param points_all: maximum trajectory length
                :return: dfsp: dataframe with padded trajectories
                """
                dfsp['ID'] = dfsp['id'].copy()
                dfsp['POINT'] = dfsp['traj_steps_all'].copy()
                ids = dfsp['ID'].unique()
                points = dfsp['POINT'].unique()
                if points_all > points.max(): points = numpy.array(range(1, points_all + 1))
                iterables = [ids, points]
                dfsp = dfsp.set_index(['ID', 'POINT'])
                mind = pandas.MultiIndex.from_product(iterables, names=['ID', 'POINT'])
                dfsp = dfsp.reindex(index=mind, fill_value=0).reset_index()
                dfsp['id'] = dfsp.groupby(dfsp['ID'])['id'].transform('max')
                dfsp['mmsi'] = dfsp.groupby(dfsp['ID'])['mmsi'].transform('max')
                dfsp['dataset_tr1_val2_test3'] = dfsp.groupby(dfsp['ID'])['dataset_tr1_val2_test3'].transform('max')
                dfsp['dataset_tr1_val2_test3'] = dfsp.groupby(dfsp['ID'])['dataset_tr1_val2_test3'].transform('max')
                dfsp = dfsp.sort_values(by=['ID', 't(t)'], ascending=[True, True]).reset_index(drop=True)
                dfsp.drop(columns=['ID', 'POINT'], inplace=True)
                return dfsp

            dfsp_norm.rename(columns=lambda x: ('%s___norm' % (x)), inplace=True)
            dfsp_pad = pad_sequences_TP_df(pandas.concat([dfsp, dfsp_norm], axis=1), points_all)
            dfsp_norm_pad = dfsp_pad[dfsp_norm.columns].copy()
            dfsp_norm_pad.rename(columns=lambda x: str(x).replace('___norm', ''), inplace=True)
            # dfsp_norm_pad.rename(columns = lambda x : str(x)[:-7], inplace=True)
            dfsp_pad = dfsp_pad[dfsp.columns].copy()
            return dfsp_pad, dfsp_norm_pad

        self.dfsp_pad, self.dfsp_norm_pad = fun_all_pad_sequences_TP_df(self.dfsp, self.dfsp_norm, points_all)

        return self.dfsp_pad, self.dfsp_norm_pad

    def fun_main_split_into_train_test_sets(self):
        """
        This function splits dataset into three different subsets: training, validation, testing
        :param dfsp_pad: dataframe with original data padded
        :param dfsp_norm_pad: dataframe with normalized/scaled data padded
        :param cfg_features_inputs: input features fed to the model
        :param cfg_features_outputs: output features predicted by the model
        :return: xy: dictionary with training(suffix -tr)/validation(suffix -va)/testing(suffix -te) subsets of
                        original & normalized(with suffix -sc) data, input(prefix x-) & output(prefix y-)
        """

        def make_3D_datasets_all(dfsp_pad, cfg_features_inputs, cfg_features_outputs):
            """
            This function transforms dataset of 2D array into 3d array, suitable for feeding RNN-based architectures
            :param dfsp_pad: datframe
            :param cfg_features_inputs: input features
            :param cfg_features_outputs: output features
            :return: xtr, xva, xte, ytr, yva, yte
            """

            def make_3D_datasets_one(dfsp_pad, cols, dataset_num):
                xytrvate = dfsp_pad[dfsp_pad['dataset_tr1_val2_test3'] == dataset_num].copy()
                return numpy.stack([xytrvate[cols] for _, xytrvate in xytrvate.groupby('id')])

            xtr = make_3D_datasets_one(dfsp_pad, cfg_features_inputs, 1)  # input training
            xva = make_3D_datasets_one(dfsp_pad, cfg_features_inputs, 2)  # input validation
            xte = make_3D_datasets_one(dfsp_pad, cfg_features_inputs, 3)  # input testing
            ytr = make_3D_datasets_one(dfsp_pad, cfg_features_outputs, 1)  # output training
            yva = make_3D_datasets_one(dfsp_pad, cfg_features_outputs, 2)  # output validation
            yte = make_3D_datasets_one(dfsp_pad, cfg_features_outputs, 3)  # output testing
            return xtr, xva, xte, ytr, yva, yte

        xy = dict()
        xy['xtr'], xy['xva'], xy['xte'], xy['ytr'], xy['yva'], xy['yte'] = make_3D_datasets_all(
            self.dfsp_pad, self.cfg_features_inputs, self.cfg_features_outputs)
        xy['xtrsc'], xy['xvasc'], xy['xtesc'], xy['ytrsc'], xy['yvasc'], xy['ytesc'] = make_3D_datasets_all(
            pandas.concat([self.dfsp_pad[['id', 'dataset_tr1_val2_test3']], self.dfsp_norm_pad], axis=1),
            self.cfg_features_inputs, self.cfg_features_outputs)
        self.xy = xy.copy()
        return self.xy

    def dataset_transformation(self):
        print("Convert series to supervised problem ....")
        t = time.time()
        dfsp = self.dataset_to_supervised_with_labels()
        print(time.time() - t)
        print("Normalise data ....")
        t = time.time()
        dfsp_norm, norm_param = self.fun_dataset_normalised()
        print(time.time() - t)
        print("Pad Sequences ....")
        t = time.time()
        dfsp_pad, dfsp_norm_pad = self.fun_dataset_padded()
        print(time.time() - t)
        print("Split dataset into train and test sets ....")
        t = time.time()
        xy = self.fun_main_split_into_train_test_sets()
        print(time.time() - t)

        return dfsp, dfsp_norm, dfsp_pad, dfsp_norm_pad, xy, norm_param, self.cfg_features_outputs


#######################################################################################################################
# # Class for Prediction & Evaluation
#######################################################################################################################
class Predictions():
    def __init__(self, best_model, xy, dfsp, dfsp_pad, norm_param, cfg_look_ahead_points, cfg_features_outputs, cfg_dataset):
        self.best_model = best_model # trained model
        self.xy = xy # input data (training, validation, testing
        self.dfsp = dfsp # dataframe containing the original data transformed to input-output (supervised problem)
        self.dfsp_pad = dfsp_pad # dataframe with original data padded
        self.norm_param = norm_param # dataframe with columns for features for normalization & rows for values mean, std, min, max
        self.cfg_look_ahead_points = cfg_look_ahead_points
        self.cfg_features_outputs = cfg_features_outputs # output features predicted by the model
        self.cfg_dataset = cfg_dataset
        
    def _fun_model_predictions(self):
        """
        This functions produces predicted outputs for training-validation-testings sets
        These predicions refer to the utm coordinates intervals [lon, lat], which are, also, in 3D format, padded and normalized
        """
        self.yhtrsc = self.best_model.predict(self.xy['xtrsc'], batch_size=1)
        self.yhvasc = self.best_model.predict(self.xy['xvasc'], batch_size=1)
        self.yhtesc = self.best_model.predict(self.xy['xtesc'], batch_size=1)

    def _fun_main_reshape_predictions(self):
        """This function reshapes 3D predicted padded outputs of training-validation-testing subsets into one 2D dataframe without padding"""
        yhsc_padded = numpy.concatenate([self.yhtrsc, self.yhvasc, self.yhtesc]).reshape(
            (self.dfsp_pad.shape[0], len(self.cfg_features_outputs)))
        yhsc_padded = pandas.DataFrame(yhsc_padded, columns=self.cfg_features_outputs)
        if self.dfsp['cumcount'].min() > 0:
            yhsc = yhsc_padded[self.dfsp_pad['cumcount'] > 0].copy()
        else:
            yhsc = yhsc_padded[self.dfsp_pad['t(t)'] > 0].copy()
        yhsc.reset_index(inplace=True, drop=True)
        self.yhsc = yhsc.copy()

    def _fun_main_denormalize_predictions(self):
        """ This function denormalizes the predicted outputs """
        self.yh_d = pandas.DataFrame()
        for x in self.norm_param.columns.values:
            cols = self.yhsc.loc[:, self.yhsc.columns.str.startswith(x)].columns.values
            for i in cols:
                self.yh_d[('%s' % (i))] = self.yhsc[('%s' % (i))] * self.norm_param.loc['std', ('%s' % (x))] + \
                                          self.norm_param.loc['mean', ('%s' % (x))]

    def _fun_main_transform_predictions(self):
        """ This function transforms predicted output of differences into actual values of lon and lat in utm & WGS84 """

        yh = self.yh_d.copy()
        yh.reset_index(inplace=True, drop=True)
        for la in range(1, self.cfg_look_ahead_points + 1):
            yh['lon(t+%d)' % (la)] = yh['dlon(t+%d)' % (la)] + self.dfsp['lon(t)'] # transform differences of lon in utm into actual values
            yh['lat(t+%d)' % (la)] = yh['dlat(t+%d)' % (la)] + self.dfsp['lat(t)'] # transform differences of lat in utm into actual values
            yh['WGS84lon(t+%d)' % (la)], yh['WGS84lat(t+%d)' % (la)] = function_pyproj(
                yh['lon(t+%d)' % (la)].values, yh['lat(t+%d)' % (la)].values, True) # transform utm coordinates to WGS84
        yh.rename(columns=lambda x: ('pred_%s' % (x)), inplace=True)
        dfsp_p = self.dfsp.copy()
        dfsp_p[yh.columns] = yh.values
        return yh, dfsp_p

    def produce_predictions(self):
        """
        This functions calls the necessary functions to produces :
        1. the final predictions: differences lon-lat in utm (dlon, dlat), actual values of lon-lat in utm & actual values of lon-lat in WGS84
        2. the dataframe containing the original data along with the final predictions
        """
        print("Predict with the best saved model (single-shot) ....")
        self._fun_model_predictions()
        print("Reshape & Denormalize & Transform predictions Ds to actual values ....")
        self._fun_main_reshape_predictions()
        self._fun_main_denormalize_predictions()
        yh_mflp, dfsp_pred = self._fun_main_transform_predictions()
        return yh_mflp, dfsp_pred


########################################################################################################################
# # # # # # # # # # # # # # # # # # # # # #   Functions for Neural Networks    # # # # # # # # # # # # # # # # # # # # #
########################################################################################################################
def build_train_simple_model(xtrsc_3d, ytrsc_3d, xvasc_3d, yvasc_3d, path_savename, file_savename, cfg_python_seed,
                             cfg_nn_model_num_epochs,
                             cfg_nn_early_stop_patience,
                             cfg_nn_model_hneurons):

    path2 = "%s/model_hdf5/" % (path_savename)
    if not os.path.exists(path2):
        os.makedirs(path2)
        print("Create new folder:", path2)

    @tf.keras.utils.register_keras_serializable()
    def dist_euclidean(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.backend.sqrt(
            tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1, keepdims=True) + 1e-16),
            axis=-1)

    class ClassTimeHistory(tf.keras.callbacks.Callback):
        """Count processing time"""

        # def init(self):
        #     self.train_loss = []
        #     self.val_loss = []
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()
            self.model.reset_states()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
            # self.train_loss.append(logs.get('loss'))
            # self.val_loss.append(logs.get('val_loss'))


    ### Set parameters for training procedure
    def make_reproducible_results_tf(cfg_python_seed):
        """ This functions set environment for reproducing results in GPU """
        os.environ['PYTHONHASHSEED'] = str(
            cfg_python_seed)  # Set `PYTHONHASHSEED` environment variable at a fixed value
        random.seed(cfg_python_seed)  # Set `python` built-in pseudo-random generator at a fixed value
        numpy.random.seed(cfg_python_seed)  # Set `numpy` pseudo-random generator at a fixed value
        tf.random.set_seed(cfg_python_seed)  # Set `tensorflow` pseudo-random generator at a fixed value
        tf.random.set_global_generator(cfg_python_seed)
        tf.keras.backend.clear_session()
        os.environ['TF_DETERMINISTIC_OPS'], os.environ['TF_CUDNN_DETERMINISTIC'] = '1', '1'

        ### Configure a new global tensorflow session
        session_conf = tf.python.ConfigProto(allow_soft_placement=True,
                                             intra_op_parallelism_threads=1,
                                             inter_op_parallelism_threads=1,
                                             device_count={'CPU': 1, 'GPU': 0},
                                             # gpu_options=tf.GPUOptions(allow_growth=True)
                                             gpu_options=tf.python.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                             )

        if tf.test.is_gpu_available():
            session_conf.device_count.update({'CPU': 1, 'GPU': 1})
            print(tf.test.gpu_device_name())  # GPU
        # session_conf = tf.Session(graph=tf.python.get_default_graph(), config=session_conf)
        # tf.config.set_visible_devices([], 'GPU')
        tf.python.keras.backend.set_session(session_conf)

    make_reproducible_results_tf(cfg_python_seed)
    checkpoint_path = ("%s/model_hdf5/%s_%s" % (path_savename, file_savename, "model-{epoch:04d}-{val_loss:.6f}.hdf5"))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=2)
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg_nn_early_stop_patience, verbose=2, mode='auto')
    history = tf.keras.callbacks.History()
    time_callback = ClassTimeHistory()
    callbacks_list = [checkpoint, earlyStopping, time_callback, history]

    ### Build & Train model
    def nn_simple_model_LSTM(xtrsc, ytrsc, xvasc, yvasc, callbacks_list, time_callback,
                             cfg_nn_model_num_epochs, cfg_nn_model_hneurons):
        """ This function builds & trains the NN model """
        start = time.time()

        """--- Design the models ---"""
        ###################################################################################################################
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(cfg_nn_model_hneurons[0], batch_input_shape=(None, None, xtrsc.shape[2]),
                       return_sequences=True, stateful=False))
        for nn0 in cfg_nn_model_hneurons[1:]: model.add(tf.keras.layers.Dense(nn0, activation='relu'))
        model.add(tf.keras.layers.Dense(ytrsc.shape[2], activation='linear'))
        ###################################################################################################################

        """--- Print model parameters ---"""
        print(model.summary())

        """--- Compile the model ---"""
        model.compile(loss=dist_euclidean, optimizer='adam', metrics=['mae', 'mse'])

        """--- Fit the model ---"""
        history = model.fit(xtrsc, ytrsc, epochs=cfg_nn_model_num_epochs, batch_size=1,
                            validation_data=(xvasc, yvasc), callbacks=callbacks_list, verbose=2)

        """--- Calculate Training Time (sec) ---"""
        print("Training time (sec):")
        end = time.time()
        print(end - start)
        times = time_callback.times
        print(sum(times))

        return history, times, model

    history, times, model = nn_simple_model_LSTM(xtrsc_3d, ytrsc_3d, xvasc_3d, yvasc_3d,
                                                 callbacks_list, time_callback,
                                                 cfg_nn_model_num_epochs,
                                                 cfg_nn_model_hneurons)

    ### Load best model
    best_model = tf.keras.models.load_model(max([(path_savename + "/model_hdf5/" + basename) for basename in
                                 [x for x in os.listdir(path_savename + "/model_hdf5/") if
                                  x.startswith(file_savename + "_model-")]], key=os.path.getmtime),
                            custom_objects={'dist_euclidean': dist_euclidean})

    ### Save best model in hdf5 format
    model.save(("%s/model_hdf5/%s_final_model.hdf5" % (path_savename, file_savename)))

    ### Evaluate training procedure
    def plot_loss_training(history, path_savename, file_savename):
        """ This functions plots loss function from training phase"""
        plt.figure()
        plt.plot(history['loss'], label='train')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.plot(history['val_loss'], label='val')
        plt.legend()
        plt.savefig(("%s/%s_%s" % (path_savename, file_savename, ("plot_loss.png"))))
        plt.show()
    plot_loss_training(model.history.history, path_savename, file_savename)

    model_training_execution_time = pandas.DataFrame(list(range(1, len(times) + 1)), columns=['epoch'])
    model_training_execution_time['time(sec)'] = pandas.DataFrame(times)  # save execution time
    model_training_execution_time['loss_tr'] = pandas.DataFrame(model.history.history['loss'])
    model_training_execution_time['loss_val'] = pandas.DataFrame(model.history.history['val_loss'])

    return best_model


#######################################################################################################################

def main_function(cfg_dataset, cfg_file_for_read, cfg_dataset_duration, cfg_sea_area, cfg_ship_type,
                  cfg_look_ahead_points, cfg_gap_period_min_max, cfg_speed_min_max, cfg_stop_iterations,
                  cfg_speed_calculate, cfg_traj_points_min_max, cfg_drop_outliers, cfg_drop_outliers_sigma,
                  cfg_drop_outliers_stop_iter, cfg_data_split_tr_va, cfg_python_seed, cfg_nn_model_num_epochs,
                  cfg_nn_early_stop_patience, cfg_nn_model_hneurons, cfg_burned_points_rnn):


    print("Create folders ....")
    def create_folder_for_save(cfg_dataset, cfg_ship_type, cfg_gap_period_min_max, cfg_speed_min_max,
                               cfg_traj_points_min_max, cfg_python_seed,
                               cfg_nn_model_num_epochs, cfg_nn_model_hneurons):
        """ This function creates the necessary folders for saving the results """
        file_savename = ("rd%d_epoch%d_h1n%d_ffn%d" % (
        cfg_python_seed, cfg_nn_model_num_epochs, cfg_nn_model_hneurons[0], cfg_nn_model_hneurons[1]))

        path_savename = ("%s/save/%s__%s/SPEED%0.2f_%d_GAP%d_%d_P%d_%d" % (
            os.getcwd(), cfg_dataset, cfg_ship_type, cfg_speed_min_max[0], cfg_speed_min_max[1],
            cfg_gap_period_min_max[1], cfg_gap_period_min_max[0], cfg_traj_points_min_max[0],
            cfg_traj_points_min_max[1]))

        if not os.path.exists(path_savename): os.makedirs(path_savename)

        return file_savename, path_savename
    file_savename, path_savename = create_folder_for_save(cfg_dataset, cfg_ship_type, cfg_gap_period_min_max, cfg_speed_min_max,
                                                          cfg_traj_points_min_max, cfg_python_seed,
                                                          cfg_nn_model_num_epochs, cfg_nn_model_hneurons)



    print("Read & Clean data ....")
    t = time.time()
    dataset = Dataset(cfg_dataset, cfg_file_for_read, cfg_dataset_duration, cfg_sea_area, cfg_ship_type,
            cfg_gap_period_min_max, cfg_speed_min_max, cfg_stop_iterations, cfg_speed_calculate, cfg_traj_points_min_max,
                      cfg_drop_outliers, cfg_drop_outliers_sigma, cfg_drop_outliers_stop_iter)
    df = dataset.load_clean_data()
    print(time.time() - t)


    print("Split & Shuffle trips ....")
    t = time.time()
    datasetpr = DatasetProcessed(df, cfg_python_seed, cfg_data_split_tr_va, cfg_gap_period_min_max, cfg_speed_calculate,
                 cfg_traj_points_min_max, cfg_burned_points_rnn)
    df = datasetpr.split_trajectories_sets_shuffle()
    print(time.time() - t)


    print("Convert series to supervised problem & Normalise & Pad & Split into train-val-test sets....")
    t = time.time()
    datasettrnsf = DatasetTransformed(df, cfg_python_seed, cfg_data_split_tr_va, cfg_look_ahead_points, cfg_gap_period_min_max, cfg_traj_points_min_max, cfg_burned_points_rnn, path_savename, file_savename)
    dfsp, dfsp_norm, dfsp_pad, dfsp_norm_pad, xy, norm_param, cfg_features_outputs= datasettrnsf.dataset_transformation()
    pandas.DataFrame({'sc_x_mean': norm_param.loc['mean'].values, 'sc_x_std': norm_param.loc['std'].values}).to_json(
        "%s/model_hdf5/%s_norm_param_mean_std.json" % (path_savename, file_savename)) # save normalized parameters
    print(time.time() - t)


    print("Build & Train LSTM model ....")
    t = time.time()
    best_model = build_train_simple_model(xy['xtrsc'], xy['ytrsc'], xy['xvasc'], xy['yvasc'],
                                          path_savename, file_savename, cfg_python_seed,
                                          cfg_nn_model_num_epochs,
                                          cfg_nn_early_stop_patience,
                                          cfg_nn_model_hneurons)
    print(time.time() - t)


    print("Produce Predictions & Evaluate Results ....")
    t = time.time()
    predictions = Predictions(best_model, xy, dfsp, dfsp_pad, norm_param, cfg_look_ahead_points, cfg_features_outputs, cfg_dataset)
    yh, dfsp_pred = predictions.produce_predictions()
    print(time.time() - t)

    print("--------  E N D  --------")



def main():

    import VLF_VRF_train_config as cfg

    cfg_python_seed = cfg.cfg_python_seed
    os.environ['PYTHONHASHSEED'] = str(cfg_python_seed)  # Set `PYTHONHASHSEED` environment variable at a fixed value
    random.seed(cfg_python_seed)  # Set `python` built-in pseudo-random generator at a fixed value
    numpy.random.seed(cfg_python_seed)  # Set `numpy` pseudo-random generator at a fixed value

    main_function(cfg.cfg_dataset, cfg.cfg_file_for_read,
                  cfg.cfg_dataset_duration, cfg.cfg_sea_area, cfg.cfg_ship_type,
                  cfg.cfg_look_ahead_points, cfg.cfg_gap_period_min_max, cfg.cfg_speed_min_max,
                  cfg.cfg_stop_iterations, cfg.cfg_speed_calculate, cfg.cfg_traj_points_min_max,
                  cfg.cfg_drop_outliers, cfg.cfg_drop_outliers_sigma, cfg.cfg_drop_outliers_stop_iter,
                  cfg.cfg_data_split_tr_va, cfg.cfg_python_seed, cfg.cfg_nn_model_num_epochs,
                  cfg.cfg_nn_early_stop_patience, cfg.cfg_nn_model_hneurons, cfg.cfg_burned_points_rnn)


if __name__ == '__main__':
    main()