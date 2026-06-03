"""
Common data filters for PV data processing. The data filters include
- nighttime filter
- clear-sky filter
- outlier filter
- variable cutoff filter
- IEC filter:
        - Irradiance [-6, 1500] W/m^2
        - Ambient temperature [-30,50] degC
        - Wind speed [0,31] m/s
        - AC power [-0.01Pnom,1.02Pnom]

Author: Lauri Karttunen.
Edited by: Väinö Anttalainen.
"""


import pandas as pd
import numpy as np
import pvlib
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression


def IEC(data, p_rated, irradiance_col_name="poa"):
    """
    Threshold conditions:
    -6 W/m2	<	irradiance	<	1500 W/m2
    -30 ∘C	<	ambient temperature	<	50 ∘C
    0 m/s	<	wind speed	<	32 m/s
    -0.01 x Pnom	<	AC power	<	1.02 x  Pnom

    Returns: Filtering mask (Pandas Series object)
    """
    
    mask = (data[irradiance_col_name] < 1500) & (data[irradiance_col_name] > -6) & (data['T'] > -30) & (data['T'] < 50) & \
           (data['wind'] > 0)
           
    if 'pv_inv_out' in data.columns:
        mask = mask & (data['pv_inv_out'] > -0.01 * p_rated) & (data['pv_inv_out'] < 1.02 * p_rated)

    return mask


def threshold(data, parameters: list, lowers=[0], uppers=[50000], negate=False):
    """
    Perform threshold filtering for given parameter.
    Args:
        lower: Lower threshold
        upper: Upper threshold
        negate: If True, negate the filter conditions
    Returns: Filtering mask (Pandas Series object)
    """
    if not isinstance(parameters, list):
        parameters = [parameters]
    if not isinstance(lowers, list):
        lowers = [lowers]
    if not isinstance(uppers, list):
        uppers = [uppers]

    mask = pd.Series(True, index=data.index)
    for parameter, lower, upper in zip(parameters, lowers, uppers):
        mask &= data[parameter].between(lower, upper)
    #mask = (lower <= data[parameter]) & (data[parameter] <= upper)
    
    if negate:
        return ~mask
    return mask


def nighttime(data, irradiance_col_name="poa"):
    """
    Points with irradiance below 5 W/m^2 are considerd night-time data.

    Returns: Filtering mask (Pandas Series object)
    """
    
    mask = data[irradiance_col_name] < 5
    return mask


def daytime(data, irradiance_col_name="poa"):
    """
    Returns: Filtering mask (Pandas Series object)
    """

    return ~nighttime(data, irradiance_col_name)


def rolling_outlier_filter(data, window_size, window_width=2, irradiance_col_name="poa", datetime_col_name='utctime'):
    """    
    Power-irradiance outliers are filtered by applying a rolling horizon filter and excluding points outside +-2 STD outside
    rolling horizon mean.

    Args:
        data (pandas DataFrame): dataframe with power and poa columns
        window_size (int): number of points withing the rolling window
        window_width (float): how many standard deviations are used as the filtering conditions

    Returns: Filtering mask (Pandas Series object)
    """

    # Sort data by irradiance to perform the rolling horizon
    data = data.sort_values(by=irradiance_col_name)

    p_poa = data['power'] / data[irradiance_col_name]

    # Set rolling window size
    window_size = window_size

    # Calculate rolling mean and standard deviation
    rolling_mean = p_poa.rolling(window=window_size).mean()
    rolling_sd = p_poa.rolling(window=window_size).std()

    # Define the upper and lower bounds
    upper_bound = rolling_mean + window_width * rolling_sd
    lower_bound = rolling_mean - window_width * rolling_sd

    p_poa_mask = (p_poa < upper_bound) & (p_poa > lower_bound)

    data['filter_mask'] = p_poa_mask
    data = data.sort_values(by=datetime_col_name)

    return data.filter_mask


def rolling_quantile_filter(data, window_size, lower_quantile=0.07, irradiance_col_name="poa", datetime_col_name='utctime'):
    """
    Filters out power-irradiance outliers using a rolling quantiles.
    
    Args:
        data (pd.DataFrame): Must contain 'power' and 'poa' columns.
        window_size (int): Size of the rolling window.
        lower_quantile (float): Lower quantile (lower bound), upper quantile will be 1 - lower_quantile.
    
    Returns: Filtering mask (Pandas Series object)
    """
    lower_q = lower_quantile
    upper_q = 1 - lower_q

    data = data.sort_values(by=irradiance_col_name)
    p_poa = data['power'] / data[irradiance_col_name]

    q_low = p_poa.rolling(window=window_size).quantile(lower_q)
    q_high = p_poa.rolling(window=window_size).quantile(upper_q)

    mask = (p_poa > q_low) & (p_poa < q_high)

    data = data.assign(filter_mask=mask).sort_values(by=datetime_col_name)
    return data['filter_mask']


def cut_outliers_ransac(data, irradiance_col_name="poa", lower=0, upper=10000, filter_threshold=10000, lower_values=True, upper_values=False):
    """
    Filters out power-irradiance outliers by fitting a line using RANSAC to upper or lower part of data and cutting the outliers.
    """
    data_range = data.loc[(data[irradiance_col_name] >= lower) & (data[irradiance_col_name] <= upper), :]
    poa = data[irradiance_col_name]
    poa_range = data_range[irradiance_col_name]#poa.loc[(poa >= lower) & (poa <= upper)]
    power = data["power"]
    power_range = data_range["power"]#power.loc[(poa >= lower) & (poa <= upper)]

    # # Get the minimum power for each unique poa value
    # data_min_power = data_range.groupby(irradiance_col_name, as_index=False)["power"].min()
    # print(data_min_power)
    
    # Extract arrays for fitting
    # poa_min = data_min_power[irradiance_col_name].to_numpy()
    # power_min = data_min_power["power"].to_numpy()

    # Step 1: bin poa and take minimum power per bin
    bins = np.linspace(poa_range.min(), poa_range.max(), 40)
    digitized = np.digitize(poa_range, bins)
    poa_min = [poa_range[digitized == i].mean() for i in range(1, len(bins))]
    power_min = [power_range[digitized == i].min() for i in range(1, len(bins))]

    poa_min = np.array(poa_min)
    power_min = np.array(power_min)

    # Step 2: Fit RANSAC on the lower-envelope points
    X = poa_min.reshape(-1, 1)
    y = power_min
    nan_mask = ~np.isnan(X.ravel()) & ~np.isnan(y)
    ransac = RANSACRegressor(#base_estimator=LinearRegression(),
                             #min_samples=0.5,
                             #residual_threshold=20, 
                             random_state=10)
    ransac.fit(X[nan_mask], y[nan_mask])

    # --- Step 3: Predict line values for all poa ---
    y_pred_all = ransac.predict(poa.to_numpy().reshape(-1, 1))

    # --- Step 4: Filtering ---
    irr_threshold = filter_threshold  # <-- set your threshold here
    mask_keep = (poa <= irr_threshold) | (power >= y_pred_all)

    # poa_filtered = poa[mask_keep]
    # power_filtered = power[mask_keep]

    # --- Plot results ---
    # line_x = np.linspace(poa.min(), poa.max(), 100).reshape(-1, 1)
    # line_y = ransac.predict(line_x)
 
    # plt.scatter(poa, power, color="lightgray", label="All data")
    # plt.plot(line_x, line_y, color="red", linewidth=2, label="RANSAC lower bound")
    # plt.axvline(irr_threshold, color="orange", linestyle="--", label="Irradiance threshold")
    # plt.scatter(poa_filtered, power_filtered, color="green", label="Kept points")
    # plt.xlabel("POA Irradiance")
    # plt.ylabel("Power")
    # plt.legend()
    # plt.show()

    return mask_keep


def clear_sky(data, latitude, longitude, altitude, threshold=0.2, datetime_col_name='utctime'):
    """
    Calculate cleasky GHI using Pvlib's haurwitz function. This model has the best performance of models which
    require only zenith angle [1]

    Args:
        data (pandas DataFrame): dataframe with ghi variable in UTC+00
        latitude (float): latitude of the system
        longitude (float): longitude of the system
        altitude (float): altitude of the system
        threshold: A percentage measured GHI-values can differ from modelled clear-sky GHI-values (float)
    
    Returns:  Filtering mask (Pandas Series object)
    
    References:
        [1] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
            Sky Models: Implementation and Analysis", Sandia National Laboratories, SAND2012-2389, 2012
    """

    temperature = 6
    
    # the function below returns a dictionary including different solar angles
    spa = pvlib.solarposition.spa_python(data[datetime_col_name], latitude, longitude,
                                         altitude=altitude, pressure=101325, temperature=temperature, delta_t=70,
                                         atmos_refract=None, how='numpy')
    
    # clearsky function returns a dataframe with clear-sky GHI values
    clear_sky = pvlib.clearsky.haurwitz(spa.apparent_zenith)
    cs_ghi = clear_sky['ghi'].values  # Clearsky ghi values
    
    # Calculate clearsky-index
    csi = data.ghi / cs_ghi
    
    return (csi >= 1.0 - threshold) & (csi <= 1.0 + threshold)


def quality_control(data):
    """
    
    """

    if "dataQC" in data.columns:
        mask = data["dataQC"] == 1
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask

def viss_day(data):
    """
    
    """

    if "viss_day" in data.columns:
        mask = data['viss_day'] == '[0, 0, 0, 0]'
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask


def viss_instant(data):
    """
    
    """

    if "viss_instant" in data.columns:
        mask = ~data['viss_instant'].isin([2, 3, 4, 5, 6])
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask
    

def snow(data):
    """
    
    """

    if "snow" in data.columns:
        mask = data["snow"] == 0
    else:
        mask = pd.Series([True] * len(data), index=data.index)

    return mask