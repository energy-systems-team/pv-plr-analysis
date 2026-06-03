"""
This file contains functions for estimating PV panel temperatures and transferring temperature data from another
dataframe.

Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen and Lauri Karttunen
"""

import math
import pandas
import config


def add_estimated_panel_temperature(df, constant_a=-3.43, constant_b=-0.125, irradiance_col_name="poa", 
                                    estimated_variable='module_temp') ->float:
    """
    Adds an estimate for panel temperature based on wind speed, air temperature and absorbed radiation.
    If air temperature, wind speed or absorbed radiation columns are missing, aborts.

    :param df: dataframe containing necessary columns 
    :param constant_a: empirical constant (see Sandia temperature model). Default values from Varjopuro et al. [1] are used
    :param constant_b: empirical constant (see Sandia temperature model). Default values from Varjopuro et al. [1] are used
    :return: module temperature in Celsius

    King 2004 model
    D.~King, J.~Kratochvil, and W.~Boyson,
    Photovoltaic Array Performance Model Vol. 8,
    PhD thesis (Sandia Naitional Laboratories, 2004).

    [1] J. Varjopuro et al., 
    Computational simulation of perovskite and silicon solar panel operating temperatures in varying ambient conditions,
    Solar Energy Materials and Solar Cells 290 (2025) 113657,
    https://doi.org/10.1016/j.solmat.2025.113657

    NOTE: Varjopuro et al. estimate the cell temperature directly from ambient temperature
    """
    # checking that all required variables exist in df

    if "T" not in df.columns:
        print("No air temperature variable in given dataframe")
        print("Aborting")
        return df

    if "wind" not in df.columns:
        print("No wind speed variable in given dataframe")
        print("Aborting")
        return df

    if irradiance_col_name not in df.columns:
        print(f"no reflection corrected poa value in df '{irradiance_col_name}'")
        print("Aborting")
        return df
    
    absorbed_radiation = df[irradiance_col_name]
    wind = df["wind"]
    module_elevation = config.module_elevation
    air_temperature = df["T"]

    # wind is sometimes given as west/east components

    # wind speed at model elevation, assumes 0 speed at ground, wind speed vector len at 2m and forms a
    # curve which describes the wind speed transition from 0 to 10m wind speed to higher
    #wind_speed = (module_elevation / 10) ** 0.1429 * wind
    wind_speed = wind
    module_temperature = absorbed_radiation * math.e ** (constant_a + constant_b * wind_speed) + air_temperature
    df[estimated_variable] = module_temperature

    return df


def add_estimated_cell_temperature(df, deltaT=1, temperature_name='module_temp', irradiance_col_name='poa_rc') ->float:
    """
    Adds an estimate for cell temperature based on module temperature, and absorbed radiation.
    If module temperature, or absorbed radiation columns are missing, aborts.
    :param df:
    :param deltaT: empirical constant (see Sandia temperature model). Default values from Varjopuro et al. [1] are used
    :param temperature_name: column name of module temperature
    :param poa_name: column name for POA irradiance
    :return:

    King 2004 model
    D.~King, J.~Kratochvil, and W.~Boyson,
    Photovoltaic Array Performance Model Vol. 8,
    PhD thesis (Sandia Naitional Laboratories, 2004).

    [1] J. Varjopuro et al., 
    Computational simulation of perovskite and silicon solar panel operating temperatures in varying ambient conditions,
    Solar Energy Materials and Solar Cells 290 (2025) 113657,
    https://doi.org/10.1016/j.solmat.2025.113657
    
    """
    
    absorbed_radiation = df[irradiance_col_name]
    module_temp = df[temperature_name]

    cell_temperature = module_temp + absorbed_radiation / 1000 * deltaT

    df["cell_temp"] = cell_temperature

    return df

def add_panel_temperature_from_cell_temperature(df, deltaT=1, cell_temp_name='cell_temp', irradiance_col_name='poa_rc'):
    """
    Calculates panel temperature from cell temperature using Sandia model.
    """

    module_temp = df[cell_temp_name] - df[irradiance_col_name] / 1000 * deltaT

    df["module_temp"] = module_temp

    return df


def add_dummy_wind_and_temp(df:pandas.DataFrame, wind=2, temp=20)-> pandas.DataFrame:
    """
    Adds dummy wind speed and air temperature values. 20 Celsius and 2 m/s wind by default.
    :param df:
    :param wind:
    :param temp:
    :return: input df with wind and air temp columns with dummy values
    """

    if "T" not in df.columns:
        df = add_dummy_temperature(df, temp)

    if "wind" not in df.columns:
        df = add_dummy_wind(df, wind)

    return df


def add_dummy_temperature(df: pandas.DataFrame, temp=20)->pandas.DataFrame :
    df["T"] = temp
    return df


def add_dummy_wind(df, wind=2):
    df["wind"] = wind
    return df


def add_wind_and_temp_to_df1_from_df2(df1: pandas.DataFrame, df2: pandas.DataFrame):
    """
    This function assumes that df2 has wind and temp info which can be transferred to df1
    :param df1: target df, pvlib generated multi day df
    :param df2: donor df, fmi open generated multi day df
    :return: target df with wind and T columns which are from df2
    """

    # If df1 does not have values where minute == 30, the frames do not align well. Can cause issues.
    # alignment issues can be avoided by using config.data_resolution of 60, 30, 15, 10, 5, 1

    # creating weather df
    weather_df = df2[["time", "wind", "T"]]

    # joining weather df to df1
    #df1 = pandas.concat([df1, weather_df], axis=1)
    df1 = df1.merge(weather_df, on="time", how="outer")

    # filling in nan values for wind
    df1['wind'] = df1['wind'].interpolate(limit_direction='both')

    # filling in nan values for temp
    df1['T'] = df1['T'].interpolate(limit_direction='both')

    df1.set_index("time", inplace=True)

    return df1



