"""
Functions in this file generate irradiance dataframes with either PVlib or FMI open.

PVlib generates clear sky irradiance estimates, meaning that the PVlib values can be used for theoretical cloud free
output estimation.

FMIopen values are weather forecast-based, and they can be used to estimate actual PV output.

Author: TimoSalola (Timo Salola).
Modified: ppkart (Lauri Karttunen)
"""


import sys
import pandas
import pandas as pd
from pvlib import location
from datetime import timedelta, datetime
import config

"""
Solar irradiance dataframe creation functions are included in this file.

Currently supports pvlib simulations and fmi open data
"""


def get_solar_irradiance(date_start: datetime, day_count:int , model="pvlib")-> pandas.DataFrame:
    """
    Returns a dataframe with datetime, ghi, dni and dhi values.
    Example output:
                                 ghi         dni        dhi
    2023-10-13 00:00:00+00:00    0.000000    0.000000   0.000000
    2023-10-13 05:00:00+00:00    0.000047    4.909022   0.000038
    2023-10-13 06:00:00+00:00   52.271574  322.948696  18.132775
    2023-10-13 07:00:00+00:00  148.793920  567.912918  33.656736
    2023-10-13 08:00:00+00:00  232.498128  682.075623  42.412783


    :param date_start, first day in model
    :param day_count: how many days to model
    :param model: string with model name, uses pvlib by default
    :return:
    """
    #print("Generating dataframe with ghi, dni, dhi using " + str(model) + ".")

    # creating interval end variable
    date_end = date_start + timedelta(days=day_count, minutes=-1)
    #print("start date:" + str(date_start) + " - " + str(date_end))

    match model:
        case "pvlib" | "pvlib_ineichen" | "inechen":
            return __get_irradiance_pvlib(date_start, date_end)
        case "pvlib_simplified_solis" | "simplified_solis" | "solis":
            return __get_irradiance_pvlib(date_start, date_end, mod="simplified_solis")
        case "meps" | "fmi_open" | "fmiopen":
            return __get_irradiance_fmiopen(date_start, date_end)


    # none of the cases activated:
    print("Error: model \"" + model + "\" not programmed into the system. Check file solar_irradiance_estimator.py")
    sys.exit(1)


def __get_irradiance_pvlib(date_start: datetime, date_end: datetime, mod="ineichen")-> pandas.DataFrame:
    """
    PVlib based clear sky irradiance modeling
    :param date: Datetime object containing a date
    :param mod: One of the 3 models supported by pvlib
    :return: Dataframe with ghi, dni, dhi. Or only GHI if using haurwitz
    """

    # creating site data required by pvlib poa
    site = location.Location(config.latitude, config.longitude, tz=config.timezone)

    # measurement frequency, for example "15min" or "60min"
    measurement_frequency = str(config.data_resolution) + "min"

    # measurement count, 1440 minutes per day
    measurement_count = 1440 / config.data_resolution

    times = pd.date_range(start=date_start,
                          end=date_end,  # year + day for which the irradiance is calculated
                          freq=measurement_frequency,  # take measurement every 60 minutes
                          tz=site.tz)  # timezone

    # creating a clear sky and solar position entities
    clearsky = site.get_clearsky(times, model=mod)

    # adds index as a separate time column, for some reason this is required as even a named index is not callable
    # with df[index_name] and df.index is not supported by function apply structures
    clearsky.insert(loc=0, column="time", value=clearsky.index)

    # returning clearsky irradiance df
    return clearsky
