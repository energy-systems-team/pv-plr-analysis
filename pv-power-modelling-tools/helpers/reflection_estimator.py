"""
This file contains functions for estimating how much of the light reaching the plane of array is lost due to
reflections.

Equations are based on Martin & Ruiz 2001 paper
"Calculation of the PV modules angular losses under field conditions by means of an analytical model"

Original author: TimoSalola (Timo Salola).
Edited by: Väinö Anttalainen
"""

import math
from datetime import datetime
import numpy
import pandas
import pandas as pd

from helpers import astronomical_calculations
import config


# panel reflectance constant, empirical value. Solar panels with better optical coatings would have a lower value where
# as uncoated panels would have a higher value. Dust on panels increases reflectance_constant.
# 0.159 is given as an average value for polycrystalline silicon module reflectance
reflectance_constant = 0.159


def add_reflection_corrected_poa_to_df(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds reflection corrected POA values, calculated from beam, diffused, and ground reflected components,
    to dataframe with name "poa_ref_cor". If measured POA is given, scales the measured value with the losses 
    calculated from components.
    :param df:
    :return:
    """
    df["poa_comp_rc"] = df["poa_beam_rc"] + df["poa_diffused_rc"] + df["poa_ground_rc"]

    if "poa" in df.columns:
        reflection_ratio = df["poa_comp_rc"] / df["poa_comp"] 
        df["poa_rc"] = df["poa"] * reflection_ratio

    return df


def add_reflection_corrected_poa_components_to_df(df: pandas.DataFrame)-> pandas.DataFrame:
    """
    BTN = dni_poa
    DTg = ghi_poa
    DT = dhi_poa
    """

    poa_beam_reflected = __poa_beam_reflected(df.index)
    poa_diffused_reflected = __poa_diffused_reflected()
    poa_ground_reflected = __poa_ground_reflected()

    df["poa_beam_rc"] = (1 - poa_beam_reflected) * df["poa_beam"]
    df["poa_diffused_rc"] = (1 - poa_diffused_reflected) * df["poa_diffused"]
    df["poa_ground_rc"] = (1 - poa_ground_reflected) * df["poa_ground"]

    return df


def __poa_beam_reflected(dt: datetime)-> float:
    """
    Computes a constant in range [0,1] which represents how much of the direct irradiance is reflected from panel
    surfaces.
    :param dt: datetime
    :return: reflected radiation in range [0,1]

    F_B_(alpha) in "Calculation of the PV modules angular losses under field conditions by means of an analytical model"
    """

    a_r = reflectance_constant

    AOI = astronomical_calculations.get_solar_angle_of_incidence_fast(dt)

    # upper section of the fraction equation
    upper_fraction = math.e ** (-numpy.cos(numpy.radians(AOI)) / a_r) - math.e ** (-1.0 / a_r)
    # lower section of the fraction equation
    lower_fraction = 1.0 - math.e ** (-1.0 / a_r)

    # fraction or alpha_BN or poa_beam_reflected
    poa_beam_reflected = upper_fraction / lower_fraction

    return poa_beam_reflected


def __poa_ground_reflected()-> float:
    """
    Computes a constant in range [0,1] which represents how much of ground reflected irradiation is reflected away from
    solar panel surfaces. Note that this is constant for an installation.
    :return: [0,1] float, 0 no light reflected, 1 no light absorbed by panels.

    F_A(beta) in "Calculation of the PV modules angular losses under field conditions by means of an analytical model"

    """

    # constants, these are from
    c1 = 4.0 / (3.0 * math.pi)

    c2 = -0.074
    a_r = reflectance_constant
    panel_tilt = numpy.radians(config.tilt)  # theta_T

    # equation parts, part 1 is used 2 times
    part1 = math.sin(panel_tilt) + (panel_tilt - math.sin(panel_tilt)) / (1.0 - math.cos(panel_tilt))

    part2 = c1 * part1 + c2 * (part1 ** 2.0)
    part3 = (-1.0 / a_r) * part2

    poa_ground_reflected = math.e ** part3

    return poa_ground_reflected


def __poa_diffused_reflected()-> float:
    """
    Computes a constant in range [0,1] which represents how much of atmospheric diffuse light is reflected away from
    solar panel surfaces. Constant for an installation. Almost a 1 to 1 copy of __poa_ground_reflected except
    "pi -" addition to part1 and "1-cos" to "1+cos" replacement in part1 as well.
    :return: [0,1] float, 0 no light reflected, 1 no light absorbed by panels.

    F_D(beta) in "Calculation of the PV modules angular losses under field conditions by means of an analytical model"
    """
    # constants

    c1 = 4.0 / (math.pi * 3.0)
    c2 = -0.074
    a_r = reflectance_constant
    panel_tilt = numpy.radians(config.tilt)  # theta_T
    pi = math.pi

    # equation parts, part 1 is used 2 times
    part1 = math.sin(panel_tilt) + (pi - panel_tilt - math.sin(panel_tilt)) / (1.0 + math.cos(panel_tilt))

    part2 = c1 * part1 + c2 * (part1 ** 2.0)
    part3 = (-1.0 / a_r) * part2

    poa_diffused_reflected = math.e ** part3

    return poa_diffused_reflected


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