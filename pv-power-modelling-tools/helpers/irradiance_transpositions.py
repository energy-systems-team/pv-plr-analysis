"""
Irradiance transposition functions. Used for transforming different solar irradiance components to panel
projected irradiance components.

Terminology:
POA: Plane of array irradiance, the total amount of radiation which reaches the panel surface at a given time. This is
the sum of direct beam component, sky diffused component, and ground reflected component.
POA = "poa_beam" + "poa_diffused" + "poa_ground"

Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen
"""

import math
import time
from datetime import datetime
import numpy
import pandas
import pandas as pd
import pvlib.irradiance
import helpers.astronomical_calculations as astronomical_calculations
import config


def print_full(x: pandas.DataFrame):
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

def irradiance_df_to_poa_df(irradiance_df:pandas.DataFrame)-> pandas.DataFrame:
    """
    This function takes an irradiance dataframe as input. This dataframe should contain ghi, dni and dhi irradiance values
    These values are then projected to the panel surfaces either using simple geometry or more complex equations.

    :param irradiance_df: Solar irradiance dataframe with ghi, dni and dhi components.
    :return: Dataframe with dni, ghi and dhi plane of array irradiance projections
    """
    
    irradiance_df["poa_beam"] = __calculate_beam_component(irradiance_df["dni"], irradiance_df.index)
    irradiance_df["poa_diffused"] = __calculate_sky_diffused_component(irradiance_df.index, irradiance_df["dhi"], irradiance_df["dni"])

    if "albedo" in irradiance_df.columns:
        print("albedo in df, using it to calculate ground reflected component")
        irradiance_df["poa_ground"] = __calculate_ground_reflected_component(irradiance_df["ghi"], irradiance_df["albedo"])
    else:
        irradiance_df["poa_ground"] = __calculate_ground_reflected_component(irradiance_df["ghi"])

    # adding the sum of projections to df as poa if it's not measured in the original data
    if "poa_comp" not in irradiance_df.columns:
        irradiance_df["poa_comp"] = irradiance_df["poa_diffused"] + irradiance_df["poa_beam"] + irradiance_df["poa_ground"]

    return irradiance_df


"""
PROJECTION FUNCTIONS
4 functions for 3 components, 2 functions for DNI as either date or angle of incidence can be used for computing the 
same result.
"""

def __calculate_beam_component(dni, dt)-> float:
    """
    Beam component of the radiation. Based on https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/plane-of-array-poa-irradiance
    /calculating-poa-irradiance/poa-beam/
    :param DNI: Direct sunlight irradiance component in W
    :param dt: Time of simulation
    :return: Direct radiation per 1m² of solar panel surface

    This version of the function is fairly well optimized.
    """

    angle_of_incidence = astronomical_calculations.get_solar_angle_of_incidence_fast(dt)

    return numpy.abs(dni * numpy.cos(numpy.radians(angle_of_incidence)))


def __project_dhi_to_panel_surface(dhi: float)-> float:
    """
    Uses atmosphere scattered sunlight and solar panel angles to estimate how much of the scattered light is radiated
    towards solar panel surfaces.
    :param dhi: Atmosphere scattered irradiation.
    :return: Atmosphere scattered irradiation projected to solar panel surfaces.
    """
    return dhi * ((1.0 + math.cos(numpy.radians(config.tilt))) / 2.0)


def __calculate_sky_diffused_component(time, dhi, dni)-> float:
    """
    Sky-diffused component of radiation. Alternative dhi model,
    Calculated internally by pvlib, pvlib documentation at:
    https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.perez.html
    """

    # function parameters
    dni_extra = pvlib.irradiance.get_extra_radiation(time)

    # this should take sun-earth distance variation into account
    # empirical constant 1366.1 should work nearly as well

    # installation angles
    surface_tilt = config.tilt
    surface_azimuth = config.azimuth

    # sun angles
    solar_azimuth, solar_zenith = astronomical_calculations.get_solar_azimuth_zenit_fast(time)

    # air mass
    airmass = astronomical_calculations.get_air_mass_fast(time)

    return pvlib.irradiance.perez(surface_tilt, surface_azimuth,dhi, dni, dni_extra,  solar_zenith, solar_azimuth, airmass, return_components=False) 


def __calculate_ground_reflected_component(ghi, albedo=config.albedo)-> float:
    """
    Ground-reflected component of the radiation. Equation from
    https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-ground-reflected/

    Uses ground albedo and panel angles to estimate how much of the sunlight per 1m² of ground is radiated towards solar
    panel surfaces.
    :param ghi: Ground reflected solar irradiance.
    :return: Ground reflected solar irradiance hitting the solar panel surface.
    """
    step1 = (1.0-math.cos(numpy.radians(config.tilt)))/2
    step2 = ghi*albedo * step1
   
    return step2 # ghi * config.albedo * ((1.0 - math.cos(numpy.radians(config.tilt))) / 2.0)


def add_albedo_from_snow_depth(df):
    """
    """
    if "snow_ground" in df.columns:
         # Step 1: create daily snow indicator (snow present if > 0)
        daily_snow_filled = df['snow_ground'].ffill(limit=1440)  # 1440 minutes in a day
        daily_snow_bool = (daily_snow_filled > 0).astype(int)

        # Step 2: shift by one day to check both current & next measurements
        snow_next_bool = daily_snow_bool.shift(freq=pd.Timedelta(days=-1)).fillna(0).astype(int)

        # Step 3: condition for confirmed snow periods
        snow_periods = (daily_snow_bool.astype(bool) & snow_next_bool.astype(bool))

        # Step 4: assign albedo = 0.7 where both days show snow
        df["albedo"] = config.albedo # Default albedo for non-snow days
        df.loc[snow_periods.reindex(df.index, method="ffill").fillna(0).astype(bool), "albedo"] = 0.7
    else:
        df["albedo"] = config.albedo
        print("No snow_ground on columns. Only default albedo added to columns.")

    print(f"albedo in irradiance transposition: {df["albedo"].unique()}")

    return df