from pvlib import irradiance
from helpers import astronomical_calculations
import numpy as np


def dni_correction(dni, utctimes, aoi=None, solar_zenith=None):
    """
    Function for implementing DNI correction from Böök DOI: 10.1016/j.solener.2020.04.068
    """
    if aoi is None:
        aoi = astronomical_calculations.get_solar_angle_of_incidence_fast(utctimes)
    if solar_zenith is None:
        solar_azimuth, solar_zenith = astronomical_calculations.get_solar_azimuth_zenit_fast(utctimes)

    # DNI correction from Böök DOI: 10.1016/j.solener.2020.04.068
    a = -838
    b = -0.112
    c = 951

    dni_qc_limit = a * np.exp(b * aoi) + c
    dni_qc = np.minimum(dni, dni_qc_limit)

    # Set DNI values with low solar elevation (under 0.5 degrees) to zero.
    # This is recommended by Böök DOI: 10.1016/j.solener.2020.04.068
    solar_elevation = 90 - solar_zenith
    dni_qc[solar_elevation < 0.5] = 0

    return dni_qc


def get_dni_and_dhi(data, overwrite=False):
    """
    Estimates or retrieves Direct Normal Irradiance (DNI) and Diffuse Horizontal Irradiance (DHI) 
    from a given dataset using the ERBS model, unless these values are already present in the data.
    It also corrects the DNI values based on work of Böök et al. (DOI: 10.1016/j.solener.2020.04.068).

    :param data: pandas.DataFrame
        A DataFrame containing at least a 'ghi' (Global Horizontal Irradiance) column and a datetime index.
        Optionally, it may also include 'dni' and/or 'dhi' columns, which will be used directly if present.
    :return: pandas.DataFrame with columns for DNI and DHI 
    """
    utctimes = data.index

    solar_azimuth, solar_zenith = astronomical_calculations.get_solar_azimuth_zenit_fast(utctimes)
    aoi = astronomical_calculations.get_solar_angle_of_incidence_fast(utctimes)

    erbs_data = irradiance.erbs(ghi=data["ghi"], 
                                  zenith=solar_zenith,
                                  datetime_or_doy=utctimes)
    dni = erbs_data["dni"]
    dhi = erbs_data["dhi"]

    dni_qc = dni_correction(dni, utctimes, aoi=aoi, solar_zenith=solar_zenith) 

    if overwrite:
        data["dni"] = dni_qc
        data["dhi"] = dhi 
    else:
        if "dni" not in data.columns:
            data["dni"] = dni_qc
        if "dhi" not in data.columns:
            data["dhi"] = dhi

    return data




