"""
This module provides automated data processing steps including filtering, calculating performance metrics, (applying
statistical models and calculating performance loss rates (PLR)).
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join('pv-power-modelling-tools'))
from helpers import filters
from helpers import output_estimator
from helpers import performance_loss_rate
import config

datetime_name = 'time'


def check_if_file_already_exists(save_path):
    if os.path.exists(save_path):
        print(save_path + ' already exists')
        return True
    return False


class DataSet:
    def __init__(self, data, datetime_name: str, power_name: str, poa_name: str, ghi_name: str, ambient_t_name: str,
                 cell_t_name: str, wind_speed_name: str, rated_power: int, latitude: float, longitude: float, altitude: float):
        """
        
        """
        # Unify the column names
        self.df = data
        self.datetime_name = datetime_name
        self.df[datetime_name] = pd.to_datetime(self.df[self.datetime_name])
        self.rated_power = rated_power  # Power at STC in W, provided by manufacturer
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.power_name = power_name
        self.poa_name = poa_name
        self.ghi_name = ghi_name
        self.ambient_t_name = ambient_t_name
        self.cell_t_name = cell_t_name
        self.wind_speed_name = wind_speed_name


    def filter(self, cellT_threshold=(-10000, 10000), iec=False, poa_threshold=(-5, 50000), nighttime=True,
               power_vs_poa=False, power_vs_poa_window_size=1000, power_vs_poa_window_width=2, clearsky=False, cs_threshold=0.2):
        """

        Args:
            cellT_threshold: Thresholds for cell temperature
            iec: (Boolean) if True, perform IEC filter
            poa_threshold: Thresholds for POA irradiance
            nighttime: (Boolean) if True, nighttime datapoints are included in data
            power_vs_poa: (Boolean) if True, perform outlier filtering based on the relation of power and POA
            clearsky: (Boolean) if True, perform clear-sky filtering
            cs_threshold: Threshold of how many percentages monitored GHI can very from modelled clear-sky GHI

        Returns: Filtered dataset

        """
        # IEC
        if iec:
            self.df = self.df[filters.IEC(self.df, self.rated_power)]
        # Nighttime
        if not nighttime:
            self.df = self.df[filters.datetime(self.df)]
        # Irradiance threshold
        self.df = self.df[filters.threshold(self.df, self.poa_name, poa_threshold[0], poa_threshold[1])]
        # cell T threshold
        self.df = self.df[filters.threshold(self.df, self.cell_t_name, cellT_threshold[0], cellT_threshold[1])]
        # P vs POA
        if power_vs_poa:
            self.df = self.df[filters.rolling_outlier_filter(self.df, power_vs_poa_window_size, power_vs_poa_window_width, irradiance_col_name=self.poa_name,
                                                             datetime_col_name=self.datetime_name)]
        if clearsky:
            self.df = self.df[filters.clear_sky(self.df, self.latitude, self.longitude, self.altitude, cs_threshold, datetime_col_name=datetime_name)]
        return self.df
    


def create_filtered_datasets(data_path: str, column_names, rated_power, filter_path: str, filter_name: str, latitude: float, longitude: float, altitude: float, 
                             iec=False, poa_threshold=(-5, 50000), nighttime=True, power_vs_poa=False, clearsky=False, cs_threshold=0.2):
    """

    Args:
        data_path:
        filter_path:
        filter_name:
        iec:
        poa_threshold:
        nighttime:
        power_vs_poa:
        clearsky:
        cs_threshold:

    Returns:

    """
    # Check if filtered data already exists
    save_path = os.path.join(filter_path, f'{filter_name}.csv')
    if check_if_file_already_exists(save_path):
        filtered = pd.read_csv(save_path, parse_dates=[datetime_name])
        return filtered

    df = pd.read_csv(data_path, parse_dates=[datetime_name])
    data = DataSet(data=df, datetime_name=column_names[datetime_name], power_name=column_names['power'], poa_name=column_names['poa'], ghi_name=column_names['ghi'],
                          ambient_t_name=column_names['T'], cell_t_name=column_names['cell_temp'], wind_speed_name=column_names['wind'], rated_power=rated_power,
                          latitude=latitude, longitude=longitude, altitude=altitude)    

    filtered = data.filter(iec=iec, poa_threshold=poa_threshold, nighttime=nighttime, power_vs_poa=power_vs_poa,
                           clearsky=clearsky, cs_threshold=cs_threshold)
    filtered.to_csv(save_path, index=False)
    
    print(save_path + ' saved')
    return filtered


def create_performance_metric_datasets(data_path: list, column_names: dict, filter_name: list, metric_path: str, metrics: dict, temp_coef: float, rated_power: int, rec_interval: int, df=None):
    """

    Args:
        data_paths: Path(s) of the dataset if df is not given as input, list or string
        column_names (dict): 
        filter_names: Description(s) of the filter (for example, PvsPOA_POA50), list of string
        metric_path:
        metrics:
        temp_coef (float): temperature coefficient of power in 1/degC (not in %/degC)
        df:

    Returns:

    """
    # If dataframe is not given as input, open a dataframe from given path location
    if df is None:
        dfs = []
        if type(data_path) == 'string':  # If only one path is given, assign the dataframe to list 'dfs'
            dfs.append(pd.read_csv(data_path, parse_dates=[datetime_name]))
        elif type(data_path) == 'list':  # If multiple paths are given, save them to list 'dfs'
            for path in data_path:
                dfs.append(pd.read_csv(path, parse_dates=[datetime_name]))
    elif type(df) != list:
        dfs = [df]
    elif type(df) == list:
        dfs = df
    for df in dfs:
        df.set_index(datetime_name, inplace=True)
    ## Create a list named 'data_sets' where DataSet-objects from dataframes in dfs are created
    #data_sets = []
    #for df in dfs:
    #    data_sets.append(Performance_metric.DataSet(df, temp_coef, rated_power, rec_interval))

    # Create a list containing the names for performance metric datasets
    # If only one filter name is given (string), save that to a list
    if type(filter_name) != list:
        filter_name = [filter_name]
    # Create a list for dataset names, which have the name of the filter in the beginning
    names = [filter_name[i] for i in range(len(filter_name))]

    poa_name = column_names['poa']

    # Loop through data list
    for i in range(len(dfs)):
        data = dfs[i]
        base_name = names[i] + '_'  # Filter part of the name
        # Loop through performance metrics that are given as inputs
        for metric, intervals in metrics.items():
            
            if metric in ['power', 'power_series']:
                # Loop through intervals
                for interval in intervals:
                    name = base_name + f'P_{interval}'
                    if check_if_file_already_exists(os.path.join(metric_path, interval, f'{name}.csv')):
                        continue
                    series = output_estimator.power_series(data, poa_name, interval)
                    _save_performance_metric(series=series, interval=interval, name=name, metric_path=metric_path)
            
            elif metric in ['pi', 'PI', 'performance index', 'performance_index_series']:
                for interval in intervals:
                    name = base_name + f'PI_{interval}'
                    if check_if_file_already_exists(os.path.join(metric_path, interval, f'{name}.csv')):
                        continue
                    series = output_estimator.performance_index_series(data, rated_power, temp_coef, poa_name, interval)
                    _save_performance_metric(series=series, interval=interval, name=name, metric_path=metric_path)
            
            elif metric in ['pr', 'PR', 'performance ratio', 'performance_ratio_series']:
                for interval in intervals:
                    name = base_name + f'PR_{interval}'
                    if check_if_file_already_exists(os.path.join(metric_path, interval, f'{name}.csv')):
                        continue
                    series = output_estimator.performance_ratio_series(data, rec_interval, rated_power, interval, poa_name)
                    _save_performance_metric(series=series, interval=interval, name=name, metric_path=metric_path)
            
            elif metric in ['prt', 'PRt', 't corrected performance ratio', 't_corrected_performance_ratio_series']:
                for interval in intervals:
                    name = base_name + f'PRt_{interval}'
                    if check_if_file_already_exists(os.path.join(metric_path, interval, f'{name}.csv')):
                        continue
                    series = output_estimator.t_corrected_performance_ratio_series(data, rec_interval, rated_power, temp_coef, interval, poa_name)
                    _save_performance_metric(series=series, interval=interval, name=name, metric_path=metric_path)
            
            elif metric in ['pvusa', 'PVUSA', 'pvusa_series']:
                for interval in intervals:
                    name = base_name + f'PVUSA_{interval}'
                    if check_if_file_already_exists(os.path.join(metric_path, interval, f'{name}.csv')):
                        continue
                    series = output_estimator.pvusa_series(data, interval, poa_name)
                    _save_performance_metric(series=series, interval=interval, name=name, metric_path=metric_path)
            
            elif metric in ['sixk', '6k', 'sixk_series', 'huld', 'Huld']:
                for interval in intervals:
                    name = base_name + f'huld_{interval}'
                    if check_if_file_already_exists(os.path.join(metric_path, interval, f'{name}.csv')):
                        continue
                    series = output_estimator.huld_series(data, rated_power, interval, poa_name)
                    _save_performance_metric(series=series, interval=interval, name=name, metric_path=metric_path)
            
            else:
                print(f'No performance metric "{metric}"')


def _save_performance_metric(series, interval, name, metric_path):
    # Change series, which is a Pandas Series object with datetimeindex, to DataFrame with datetime column
    df = pd.DataFrame({datetime_name: series.index, name.split('_')[-2]: series.values})
    df.to_csv(os.path.join(metric_path, interval, f'{name}.csv'), index=False)
    print(os.path.join(metric_path, interval, f'{name}.csv') + ' saved')


def create_filtered_and_perf_metric_datasets(data_path: str, column_names: dict, rated_power: int,  temp_coef: float, rec_interval: int, filter_path: str, filter_name: str, metric_path: str,
                                             metrics: dict, latitude: float, longitude: float, altitude: float, iec=False, poa_threshold=(-5, 50000), nighttime=True,
                                             power_vs_poa=False, clearsky=False, cs_threshold=0.2):
    """

    Args:
        data_path: Filepath of the dataset original dataset
        column_names:
        rated_power: in Wp
        temp_coef: in %/degC
        rec_interval: in minutes
        filter_path: Directory path for saving the filtered dataset
        filter_name: Save name of filtered dataset
        metric_path: Directory path for performance metric dataset
        metric_name: Name of performance metric dataset
        metrics: Dictionary where keys are metrics that are calculated (power, pi, pr, prt, pvusa, sixk) and values
                are intervals ('None', 'D', 'W', 'M')
        iec:
        poa_threshold:
        nighttime:
        power_vs_poa:
        clearsky:
        cs_threshold:

    """
    filtered = create_filtered_datasets(data_path=data_path, column_names=column_names, rated_power=rated_power, filter_path=filter_path, filter_name=filter_name, 
                                        latitude=latitude, longitude=longitude, altitude=altitude, iec=iec, poa_threshold=poa_threshold, nighttime=nighttime, 
                                        power_vs_poa=power_vs_poa, clearsky=clearsky, cs_threshold=cs_threshold)
    create_performance_metric_datasets(data_path='', column_names=column_names, filter_name=filter_name, metric_path=metric_path, metrics=metrics, df=filtered, temp_coef=temp_coef,
                                       rated_power=rated_power, rec_interval=rec_interval)
