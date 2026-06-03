"""
Functions for calculating performance loss rates of given performance time series dataset. 
Methods: least squares linear regression, robust regression (Huber loss), and year-on-year.

Original author: Lauri Karttunen
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from rdtools import degradation_year_on_year
import statsmodels.api as sm
from csv import writer
import re
import os
import matplotlib.pyplot as plt



def _write_csv(filename, folder, slope, intercept, plr, uplr, name, method, POA_u, POA_l, set_POA_thresholds=False):
    """
    Function for saving PLR results to a csv file in a specified file location
    """
    desc = name
    slope = slope
    intercept = intercept
    uncertainty = uplr
    plr = plr
    method = method
    # Initialize the variables that are not part of every description
    CS = np.nan
    day = 0
    poa_upper = np.nan
    PvsG = 0
    desc_parts = desc.split('_')  # Split description part to list
    for part in desc_parts:
        if re.search('^POA[0-9]*', part):
            poa_lower = int(part[3:])
        elif re.match('CS', part):
            CS = part
        elif re.match('PvsPOA', part):
            PvsG = 1
        elif re.search('day', part):
            day = 1
        elif part in ['insta', 'D', 'ME', 'W']:
            freq = part
        elif part in ['PR', 'PRt', 'P', 'PVUSA', 'PI', 'huld']:
            metric = part
    if day == 1 and 'poa_lower' in locals():
        poa_lower = max(poa_lower, 5)
    if set_POA_thresholds:  # Overwrite threshold values if selected True
        poa_upper = POA_u
        poa_lower = POA_l

    new_row = [desc, poa_lower, poa_upper, CS, PvsG, day, metric, freq, method, slope, intercept, plr, uncertainty]

    filename = filename

    with open(os.path.join(folder, filename), 'a', newline='') as f:
        writer_object = writer(f)
        writer_object.writerow(new_row)

        # Close the file object
        f.close()


def _plr_uncertainty(slope, intercept, slope_error, intercept_error):
    """
    Function for calculating the uncertainty of PLR calculated by linear regression
    """
    ua = slope_error
    ub = intercept_error

    uplr = np.sqrt((1 / intercept) ** 2 * ua ** 2 + (-1 * slope / intercept ** 2) ** 2 * ub ** 2)
    return 1.96 * uplr  # 95% confidence interval


def linear_regression(filename, folder, df=None, filepath=None, post_filter=None, name=None, POA_u=None, POA_l=None, set_POA_thresholds=False):
    """
    Function for calculating PLR with least-squeares linear regression

    Inputs:
        filename (str):
        folder (str):
        df (None or pandas Dataframe): default None
        filepath (None or str): default None
        post_filter (None or float): default None. If float is given, remove all performance points below this float when calculating PLR.
        name (None or str): default None
        POA_u (None or float): default None
        POA_l (None or float): default None
        set_POA_threshold (Boolean): default None 
    """

    if filepath is not None:
        df = pd.read_csv(filepath, parse_dates=['time'])
        name = filepath.split('\\')[-1].split('.')[0]  # Get the name of the csv file

    if post_filter is not None:
        df = df[df.iloc[:, -1] >= post_filter] 
    
    # Drop infinite and nan values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if len(df) == 0:
        return
    
    # x is the elapsed years as floats
    x = (df['time'] - df['time'].iloc[0]) / pd.Timedelta('365 days')
    y = np.array(df.values)[:, 1].astype(np.float32)  # Df has 2 columns, datetimes and performance data, select the latter
    res = linregress(x, y)

    # Get the PLR and the uncertainty
    plr = res.slope / res.intercept
    uplr = _plr_uncertainty(res.slope, res.intercept, res.stderr, res.intercept_stderr)

    # Write the results to a file
    _write_csv(filename, folder, slope=res.slope, intercept=res.intercept, plr=plr, uplr=uplr, name=name, method='OLSLR', POA_u=POA_u,
               POA_l=POA_l, set_POA_thresholds=set_POA_thresholds)


def robust_regression(filename, folder, df=None, filepath=None, post_filter=None, name=None, POA_u=None, POA_l=None, set_POA_thresholds=False):
    """
    Function for calculating PLR with robust regression
    
    Inputs:
        filename (str):
        folder (str):
        df (None or pandas Dataframe): default None
        filepath (None or str): default None
        post_filter (None or float): default None. If float is given, remove all performance points below this float when calculating PLR.
        name (None or str): default None
        POA_u (None or float): default None
        POA_l (None or float): default None
        set_POA_threshold (Boolean): default None 
    """

    if filepath is not None:
        df = pd.read_csv(filepath, parse_dates=['time'])
        name = filepath.split('\\')[-1].split('.')[0]

    if post_filter is not None:
        df = df[df.iloc[:, -1] >= post_filter]

    # Drop infinite and nan values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if len(df) == 0:
        return

    # x is the elapsed years as floats
    x = ((df['time'] - df['time'].iloc[0]) / pd.Timedelta('365 days')).values.reshape(-1, 1)
    X = np.hstack((x, np.atleast_2d(np.ones(len(x))).T))  # Add constant (1) column to X to obtain intercept
    y = np.array(df.values)[:, 1].astype(np.float32)  # Df has 2 columns, datetimes and performance data, select the latter

    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    try:
        res = rlm_model.fit()
    except ZeroDivisionError:
        print(f'Failed with RL attempt with {name} (ZeroDivisionError)')
        return

    slope = res.params[0]
    intercept = res.params[1]

    # Get the PLR and the uncertainty
    plr = slope / intercept
    slope_error = res.bse[0]
    intercept_error = res.bse[1]
    uplr = _plr_uncertainty(slope, intercept, slope_error, intercept_error)

    # Write the results to a file
    _write_csv(filename, folder, slope=slope, intercept=intercept, plr=plr, uplr=uplr, name=name, method='RLR',
               POA_u=POA_u, POA_l=POA_l, set_POA_thresholds=set_POA_thresholds)


def yoy(filename, folder, df=None, filepath=None, post_filter=None, name=None, POA_u=None, POA_l=None, set_POA_thresholds = False):
    """
    Function for calculating PLR with year-on-year method.
    
    Inputs:
        filename (str):
        folder (str):
        df (None or pandas Dataframe): default None
        filepath (None or str): default None
        post_filter (None or float): default None. If float is given, remove all performance points below this float when calculating PLR.
        name (None or str): default None
        POA_u (None or float): default None
        POA_l (None or float): default None
        set_POA_threshold (Boolean): default None 
    """

    if filepath is not None:
        df = pd.read_csv(filepath, parse_dates=['time'])
        name = filepath.split('\\')[-1].split('.')[0]
        
    if post_filter is not None:
        df = df[df.iloc[:, -1] >= post_filter]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if len(df) == 0:
        return
    
    df = df.set_index('time')
    df = df.set_axis(['performance'], axis=1)
    data = df.performance.astype(np.float32)

    try:
        yoy_rd, yoy_ci, yoy_info = degradation_year_on_year(data, confidence_level=95, recenter=True)
    except ValueError:
        print(f'Failed yoy attempt with {name}')
        return

    # Get the PLR and the uncertainty
    uplr = (max(yoy_ci)-min(yoy_ci))/2/100
    #uplr = np.nan
    plr = yoy_rd / 100
    slope = np.nan
    intercept = np.nan

    # Write the results to a file
    _write_csv(filename, folder, slope=slope, intercept=intercept, plr=plr, uplr=uplr, name=name, method='YOY',
               POA_u=POA_u, POA_l=POA_l, set_POA_thresholds=set_POA_thresholds)

