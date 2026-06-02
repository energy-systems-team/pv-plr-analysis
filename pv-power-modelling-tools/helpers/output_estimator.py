"""
This file contains functions for estimating PV system output using Huld et. al 2010 PV output model
Required input is a dataframe with columns which contain absorbed radiation and panel temperature.
These columns are named "poa_ref_cor"(plane of array irradiance with reflection corrections) and
"module_temp" for PV module temperature.
 
Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen and Lauri Karttunen
"""

import pandas as pd
import numpy as np
import datetime
import numpy
from scipy.optimize import curve_fit
from functools import partial
import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
 
 
def print_full(x: pd.DataFrame):
    """
    Prints a dataframe without leaving any columns or rows out. Useful for debugging.
    """
 
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1400)
    pd.set_option('display.float_format', '{:10,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
   

def estimate_huld(rated_power, data, radiation_col_name="poa_rc", power_col_name="power", interval="insta", training_year=0):
    """
    Huld regression model is a predictive model using module temperature and in-plane irradiance [1].
    Authors fit the model to indoor data but here outdoor data is used. Coefficients are obtained from
    one training year, after which performance losses can be tracked by comparing the measured powers to
    predicted ones.
    Args:
        data (Pandas DataFrame): input dataframe
        interval (str): 'insta' (no aggregation), 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly) NOTE currently not called
        training_year (int): The number of the year the data is trained. If 0, default coefficients k1-k6 will be used.
    Returns:
 
    References:
        [1] Thomas Huld et al. A power-rating model for crystalline silicon PV modules,
        Solar Energy Materials and Solar Cells, Volume 95, Issue 12, 2011, Pages 3359-3369, ISSN 0927-0248,
        https://doi.org/10.1016/j.solmat.2011.07.026.
    """
 
    # data = data.dropna()  # Drop nan values rowwise to be able to perform the fitting
 
    # If module temperature column not in data, use cell temperature
    if 'module_temp' not in data.columns:
        data['module_temp'] = data['cell_temp'].copy()

    # huld 2010 constants
    k1 = -0.017162
    k2 = -0.040289
    k3 = -0.004681
    k4 = 0.000148
    k5 = 0.000169
    k6 = 0.000005

    default_coeffs = [k1, k2, k3, k4, k5, k6]

    # If training years are given, fit the coefficients. Otherwise, use the default coefficients for predictions.
    if training_year > 0:
        training_data = data[data.index <= data.index[0] + datetime.timedelta(days=365*training_year)]
        testing_data = data[data.index > data.index[0] + datetime.timedelta(days=365*training_year)]

        print(f"Training data len: {training_data.shape[0]}")

        i, t, p = (training_data[radiation_col_name]).values.astype('float64'), \
                (training_data['module_temp']).values.astype('float64'), \
                (training_data[power_col_name]).values.astype('float64')
        # Fit to data and find the coefficients
        #coeffs, pcov = curve_fit(partial(_huld, config.rated_power*1000), [i, t], p)
        #coeffs, pcov = curve_fit(f=_huld,
        #                         xdata=[config.rated_power*1000, [i, t]],
        #                         ydata=p,
        #                         method="lm")
        coeffs, pcov = curve_fit(f=partial(_huld, rated_power), 
                                 xdata=[i, t], 
                                 ydata=p,
                                 p0=default_coeffs,
                                 method="lm")

        print(f"Huld fitted coefficients (scaled): {numpy.array(coeffs) / rated_power}")
    else:
        coeffs = default_coeffs
        testing_data = data
        
    # Make the predictions for the whole dataset (including the years used for fitting)
    pred_p = _huld(rated_power, [testing_data[radiation_col_name], testing_data['module_temp']],
                   coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5])
   
    # # Calculate the performance points by dividing output power with the predicted power
    # norm_p = data.power / pred_p.values
    # 
    # return G_weighted_aggregation(interval, data['poa'], norm_p)
    return pred_p


def huld_series(data, p_rated, interval, radiation_col_name="poa_rc", training_year=0):
    """
    Huld regression model is a predictive model using module temperature and in-plane irradiance [1]. 
    Authors fit the model to indoor data but here outdoor data is used. Coefficients are obtained from 
    one training year, after which performance losses can be tracked by comparing the measured powers to 
    predicted ones.
    Args:
        data (Pandas DataFrame): input dataframe
        p_rated (float): nameplate power rating of the PV system in watts
        interval (str): 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly)
        training_year (int): The number of the year the data is trained
    Returns:

    References:
        [1] Thomas Huld et al. A power-rating model for crystalline silicon PV modules,
        Solar Energy Materials and Solar Cells, Volume 95, Issue 12, 2011, Pages 3359-3369, ISSN 0927-0248,
        https://doi.org/10.1016/j.solmat.2011.07.026.
    """

    data = data.dropna()  # Drop nan values rowwise to be able to perform the fitting

    # If module temperature column not in data, use cell temperature
    if 'module_temp' not in data.columns:
        data['module_temp'] = data['cell_temp'].copy()

    training_data = data[data.index <= data.index[0]+datetime.timedelta(days=365)]
    i, t, p = (training_data[radiation_col_name]).values.astype('float64'), \
              (training_data['module_temp']).values.astype('float64'), \
              (training_data['power']).values.astype('float64')
    # Fit to data and find the coefficients

    coeffs, pcov = curve_fit(partial(_huld, p_rated), [i, t], p)

    # Make the predictions
    pred_p = _huld(p_rated, [data[radiation_col_name], data['module_temp']], 
                   coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5])
    
    # Calculate the performance points by dividing output power with the predicted power
    norm_p = data.power / pred_p.values

    return G_weighted_aggregation(interval, data[radiation_col_name], norm_p)


def _huld(rated_power, X, k1, k2, k3, k4, k5, k6):
    """
    Args:
        p_rated (float): nameplate power rating of the PV system in watts
        X:
        k1-k6:
    Returns:
    """
    ref_temp = 25
    ref_irrad = 1000
 
    i, t = X
    G = i / ref_irrad
    G[G <= 0] = 0.000001 # Make zeros small floats for log
    T = t - ref_temp
    pred_p = G*(rated_power + k1*numpy.log(G) + k2*numpy.log(G)**2 + k3*T + k4*T*numpy.log(G) + k5*T*numpy.log(G)**2 + k6*T**2)
   
    return pred_p


def estimate_pvwatts(rated_power, data, radiation_col_name="poa_rc", temp_coef = -0.004)-> float:
    """
 
    :param rated_power:
    :param data:
    :radiation_col_name:
    :temp_coef:
    :return: Estimated system output in watts.
    """
    absorbed_radiation = data[radiation_col_name]
    c = 1 + temp_coef * (data['cell_temp'] - 25.0)
    output = c * rated_power * absorbed_radiation / 1000.0
 
    return output

 
def estimate_pvusa(data, radiation_col_name="poa_rc", power_col_name="power", training_year=0):
    """
    
    :param data:
    :radiation_col_name:
    :training_year:
    :return: Estimated system output in watts.
    """

    # If module temperature column not in data, use cell temperature
    if 'module_temp' not in data.columns:
        data['module_temp'] = data['cell_temp'].copy()

    # Coefficients from conference paper by Daryl Myers
    # "Evaluation of the Performance of the PVUSA Rating Methodology 
    # Applied to DUAL Junction PV Technology" (2009)
    a = 1.41870
    b = 0.000051
    c = 0.002291
    d = 0.000361

    # Coefficients for Helsinki's system (from Väinö's master thesis)
    # a = 14.1870
    # b = 0.001288
    # c = 0.107726
    # d = 0.159385
    
    default_coeffs = [a, b, c, d]

    # If training years are given, fit the coefficients. Otherwise, use the default coefficients for predictions.
    if training_year > 0:
        training_data = data[data.index <= data.index[0] + datetime.timedelta(days=365*training_year)]

        print(f"Training data len: {training_data.shape[0]}")

        #i, t, p = (training_data[radiation_col_name]).values.astype('float64'), \
        #        (training_data['module_temp']).values.astype('float64'), \
        #        (training_data[power_col_name]).values.astype('float64')
    
        i, t, w, p = (training_data[radiation_col_name]).values.astype('float64'), \
                    (training_data['T']).values.astype('float64'), \
                    (training_data['wind']).values.astype('float64'), \
                    (training_data[power_col_name]).values.astype('float64')
    
        # Fit to data and find the coefficients
        coeffs, pcov = curve_fit(f=_pvusa, 
                                 xdata=[i, t, w], 
                                 ydata=p,
                                 p0=default_coeffs,
                                 method="lm")

        print(coeffs)
    else:
        coeffs = default_coeffs
        
    # Make the predictions for the whole dataset (including the years used for fitting)
    pred_p = _pvusa([data[radiation_col_name], data['T'], data['wind']], coeffs[0], coeffs[1], coeffs[2], coeffs[3])
   

    return pred_p


def estimate_general_and_finetuned_pvusa(rated_power_list, data_list, system_name_list, power_col_name="power", training_years=0):
    for i in range(len(rated_power_list)):
        data = data_list[i]

        scaled_power_col_name = f"scaled_{power_col_name}"
        # Compute scaled_power and POA if not present
        if scaled_power_col_name not in data.columns:
            data[scaled_power_col_name] = data[power_col_name] / rated_power_list[i]
        if "poa_rc" not in data.columns and "poa_comp_rc" in data.columns:
            data["poa_rc"] = data["poa_comp_rc"]

    for i in range(len(rated_power_list)):
        print(f"\nProcessing system: {system_name_list[i]}")
        test_data = data_list[i]
        train_data = pd.concat([data_list[j] for j in range(len(data_list)) if j != i])

        irr, temp, wind, power = (train_data["poa_rc"]).values.astype('float64'), \
                    (train_data['T']).values.astype('float64'), \
                    (train_data['wind']).values.astype('float64'), \
                    (train_data[scaled_power_col_name]).values.astype('float64')
        
        # Coefficients from conference paper by Daryl Myers
        # "Evaluation of the Performance of the PVUSA Rating Methodology 
        # Applied to DUAL Junction PV Technology" (2009)
        a = 1.41870
        b = 0.000051
        c = 0.002291
        d = 0.000361

        default_coeffs = [a, b, c, d]

        # Fit to train data and find the general coefficients 
        # NOTE: the power is scaled with rated power
        general_coeffs, pcov = curve_fit(f=_pvusa, 
                                 xdata=[irr, temp, wind], 
                                 ydata=power,
                                 p0=default_coeffs,
                                 method="lm")
        
        print(f"PVUSA general coefficients for {system_name_list[i]}: {general_coeffs}")
        
        pred_p_general = _pvusa([test_data["poa_rc"], 
                                 test_data['T'], 
                                 test_data['wind']], 
                                 general_coeffs[0], general_coeffs[1], general_coeffs[2], general_coeffs[3])

        # Add prediction to test data
        data_list[i]["PVUSA_gen"] = pred_p_general * rated_power_list[i]

        # Optional fine-tuning
        if training_years > 0:
            print(f"Fine-tuning model for {training_years} year(s)...")
            fine_tune_start = test_data.index.min()
            fine_tune_end = fine_tune_start + datetime.timedelta(days=365 * training_years)

            fine_train = test_data[(test_data.index >= fine_tune_start) & (test_data.index < fine_tune_end)]
            fine_test = test_data[(test_data.index >= fine_tune_end)]

            irr, temp, wind, power = (fine_train["poa_rc"]).values.astype('float64'), \
                    (fine_train['T']).values.astype('float64'), \
                    (fine_train['wind']).values.astype('float64'), \
                    (fine_train[scaled_power_col_name]).values.astype('float64')

            # Reuse previously gotten coeffs and use them as default
            finetuned_coeffs, pcov = curve_fit(f=_pvusa, 
                                 xdata=[irr, temp, wind], 
                                 ydata=power,
                                 p0=general_coeffs,
                                 method="lm")
            
            print(f"PVUSA fine-tuned coefficients for {system_name_list[i]}: {finetuned_coeffs}")

            pred_p_finetuned = _pvusa([fine_test["poa_rc"], 
                                       fine_test['T'], 
                                       fine_test['wind']], 
                                       finetuned_coeffs[0], finetuned_coeffs[1], finetuned_coeffs[2], finetuned_coeffs[3])
            
            data_list[i]["PVUSA_fit"] = pd.Series(index=test_data.index, dtype=float)
            data_list[i].loc[fine_test.index, "PVUSA_fit"] = pred_p_finetuned * rated_power_list[i]

    return data_list


def _pvusa(X, a, b, c, d):
    """
    Args:
        X:
        a-d:
    Returns:
    """
    i, t, w = X
    return i * (a + b * i + c * w + d * t)


def power_series(data, radiation_col_name="poa_rc", interval='insta'):
    """
    Args:
        data (Pandas DataFrame): input dataframe 
        interval (str): 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly)
    Returns: Pandas Series object
    """
    if interval == 'insta':
        return data.power
    
    return G_weighted_aggregation(interval, data[radiation_col_name], data.power)


def performance_index_series(data, p_rated, gamma, radiation_col_name="poa_rc", interval='insta'):
    """
    Performance index (PI) is normalized, irradiance and temperature corrected power. It is very similar to
    temperature corrected PR but instead of using energy yields, power values are used.
    Args:
        data (Pandas DataFrame): input dataframe
        p_rated (float): nameplate power rating of the PV system in watts
        gamma (float): temperature coefficient of power in 1/degC (not %/degC)
        interval (str): 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly)
    Returns: Pandas dataframe object including power and datetime
    """
    ref_temp = 25
    ref_irrad = 1000

    pi = data.power / (p_rated * (data[radiation_col_name] / ref_irrad) * (1 + gamma * (data.cell_temp - ref_temp)))
    
    if interval == 'insta':
        return pi
    
    return G_weighted_aggregation(interval, data[radiation_col_name], pi)


def performance_ratio_series(data, data_resolution, p_rated, interval, radiation_col_name="poa_rc"):
    """
    One of the most common performance metrics is the performance ratio (PR), which gives the
    ratio of a system's final yield to its reference yield from a given time period, e.g. one day.
    Args:
        data (Pandas DataFrame): input dataframe
        data_resolution (int): time resolution of data in minutes 
        interval (str): 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly)
    Returns:
    """
    ref_irrad = 1000
    energy_norm = data.power * (data_resolution/60) / p_rated  # resolution/60 to get Wh 
    insolation_norm = data[radiation_col_name] * (data_resolution/60) / ref_irrad
    
    return energy_norm.resample(interval).sum() / insolation_norm.resample(interval).sum()


def t_corrected_performance_ratio_series(data, data_resolution, p_rated, gamma, interval, radiation_col_name="poa_rc"):
    """
    This version of PR takes the temperature dependency of power into account using the maximum power temperature
    coefficient.
    Args:
        data (Pandas DataFrame): input dataframe
        data_resolution (int): time resolution of data in minutes 
        interval (str): 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly)
    Returns:
    """
    ref_temp = 25
    ref_irrad = 1000
    energy_norm = data.power * (data_resolution/60) / p_rated  # resolution/60 to get Wh 
    insolation_norm_tcor = data[radiation_col_name] * (data_resolution/60) / ref_irrad * (1 + gamma*(data.cell_temp - ref_temp))
    
    return energy_norm.resample(interval).sum() / insolation_norm_tcor.resample(interval).sum()


def pvusa_series_extrapolate_to_PTC(self, interval):
    """
    PVUSA rating method is based on a regression model P=POA*(a + b*POA + c*W + d*T). Here, PVUSA is applied by
    fitting the regression model separately to each interval and using the coefficients a-d obtained
    from each fit to calculate monthly power values at PVUSA Test Conditions (POA=1000W/m2, T=20degC, W=1m/s).
    Performance losses can be tracked by comparing these values at PTC.
    
    Returns:
    """
    # Create a copy of df with only required variables
    data = self.df[['datetime', 'POA', 'TEMP', 'windSpeed', 'power']].copy()
    data.dropna(inplace=True)  # Drop nan values rowwise to be able to perform the fitting
    # Split this dataframe to interval blocks, which are saved to a list
    intervals = [g for n, g in data.set_index('datetime', drop=False).groupby(pd.Grouper(freq=interval))]
    # Create a df for coefficients
    coeffs = pd.DataFrame(columns=['datetime', 'a', 'b', 'c', 'd'])
    # Iterate through intervals
    for interval in intervals:
        i, t, w, p = (interval.POA).astype('float64'), (interval.TEMP).astype('float64'), (interval.windSpeed).astype('float64'), (interval.power).astype('float64')  # Rename variables
        try:
            popt, pcov = curve_fit(self._pvusa, [i, t, w], p)  # Find coefficients (popt)
            coeffs.loc[len(coeffs.index)] = [interval.index[-1]] + list(popt)  # Add coefficients to df
        except:
            continue
    # Calculate power at PTC
    ptc_power = coeffs.apply(lambda row: self._pvusa([self.REF_G, self.REF_T_PVUSA, self.REF_WS], row[1], row[2],
                                                     row[3], row[4]), axis=1)
    ptc_power = pd.DataFrame({'ptc_power': ptc_power}).set_index(coeffs.datetime)  # Create a df to change indexes
    return ptc_power.ptc_power
    

def pvusa_series(data, interval, radiation_col_name="poa_rc", training_year=0):
    """
    PVUSA rating method is based on a regression model P=POA*(a + b*POA + c*W + d*T) [1]. Here, PVUSA is applied by
    fitting the regression model to first year of data and using the coefficients a-d obtained
    from the fit to estimate the power.
    Args:
        interval (str): 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly)
        training_year (int): The number of the year the data is trained
    Returns:
    References:
        [1] C.M. Whitaker, T.U. Townsend, J.D. Newmiller, D.L. King, W.E. Boyson, J.A. Kratochvil, D.E. Collier, D.E. Osborn
        Application and validation of a new PV performance characterization method
        Conf. Rec. 26th IEEE Photovolt. Spec. Conf. (1997), pp. 1253-1256, 10.1109/pvsc.1997.654315
    """
    data = data.dropna()  # Drop nan values rowwise to be able to perform the fitting
    
    # Split this dataframe to yearly blocks, which are saved to a list
    #years = [g for n, g in data.groupby(pd.Grouper(freq='YE'))]
    ## Rename variables
    #i, t, w, p = (years[training_year]['poa']).values.astype('float64'), \
    #             (years[training_year]['T']).values.astype('float64'), \
    #             (years[training_year]['wind']).values.astype('float64'), \
    #             (years[training_year]['power']).values.astype('float64')
    training_data = data[data.index <= data.index[0]+datetime.timedelta(days=365)]
    i, t, w, p = (training_data[radiation_col_name]).values.astype('float64'), \
                 (training_data['T']).values.astype('float64'), \
                 (training_data['wind']).values.astype('float64'), \
                 (training_data['power']).values.astype('float64')

    # Fit to data and find the coefficients
    coeffs, pcov = curve_fit(_pvusa, [i, t, w], p)

    # Make the predictions
    pred_p = _pvusa([data[radiation_col_name], data['T'], data['wind']], coeffs[0], coeffs[1], coeffs[2], coeffs[3])

    # Calculate the performance points by dividing output power with the predicted power
    norm_p = data.power / pred_p.values

    return G_weighted_aggregation(interval, data[radiation_col_name], norm_p)
    

def _pvusa(X, a, b, c, d):
    """
    Args:
        X:
        a-d:
    Returns:
    """
    i, t, w = X
    return i * (a + b * i + c * w + d * t)


def G_weighted_aggregation(interval, irradiance_series, metric):
    """
    Calculate the irradiance-weighted average values of given time interval.
    Args:
        interval: string-object, 'D' (Daily), 'W' (weekly), 'ME' (monthly) or 'Y' (yearly)
        irradiance_series: 
        metric: pandas Series of performance metric to be aggregated
    Returns: Aggregated series
    """
    return (irradiance_series * metric).resample(interval).sum() / irradiance_series.resample(interval).sum()
