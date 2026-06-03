"""
This file contains functions for pre-processing data.

Original author: Väinö Anttalainen.
Other editors: Lauri Karttunen
"""

import pandas as pd
import numpy as np

def merge_measurements(df, col_names, diff=10):
    """
    Function for merging two measured variables.
    Checks if the difference between the measurements is less than diff and then averages them.
    Otherwise returns nan.
    :param df:
    :param col_names:
    :param diff:
    :return: 
    """
    measurement_1 = df[col_names[0]]
    measurement_2 = df[col_names[1]]

    measurements_df = pd.DataFrame({'measurement_1': measurement_1, 'measurement_2': measurement_2})

    # Calculate the average where the absolute difference is less than diff
    measurements_df['avg'] = np.where(
            np.abs(measurements_df['measurement_1'] - measurements_df['measurement_2']) < diff, 
            measurements_df[['measurement_1', 'measurement_2']].mean(axis=1), np.nan)

    return measurements_df['avg']


def impute_columns(df, col_names, resolutions):
        """
        Function for imputing missing values due to the different measurement resolutions.
        :param df:
        :param col_names: List of column names that are imputed.
        :param resolution: List of measurement resolutions for the given columns in minutes.
        :return: The modified df.
        """
        if len(col_names) != len(resolutions):
                print("Length of col_names and resolutions must match. Columns not imputed.")
        else:
                num_of_rows_to_impute_list =  [res - 1 for res in resolutions]
                for i in range(len(col_names)):
                        num_of_rows_to_impute = num_of_rows_to_impute_list[i]
                        column = col_names[i]
                        values = df[column].values
                        non_na_indices = np.where(~pd.isna(values))[0]
                        ind_prev = 0
                        for ind in non_na_indices:
                                start_ind = max(0, ind - num_of_rows_to_impute) 
                                if ind - ind_prev >= num_of_rows_to_impute:
                                        values[start_ind:ind] = values[ind]
                                ind_prev = ind
                        
                        df[column] = values
        return df


def calculate_power_withour_plr_losses(df, plr_value, initial_date):
        """
        Function for calculating what the power values would be without PLR losses (%/year).

        "real version"
        power_without_plr_losses = power_with_plr_losses / (1 - plr_value)^years_from_initial_date

        "simplified version"
        power_without_plr_losses = power_with_plr_losses / (1 - plr_value * years_from_initial_data)

        1 minute is 1/(60*24*365) = 1/525600th of a year.
        """

        initial_date = pd.to_datetime(initial_date)

        mins_in_year = 60*24*365
        mins_in_day = 24*60
        power_with_plr_losses = df["power"]

        date_difference = df.index - initial_date
        mins_from_beginning = date_difference.days * mins_in_day + np.floor(date_difference.seconds / 60)

        power_without_plr_losses = power_with_plr_losses / (1 - np.abs(plr_value)/100 * (mins_from_beginning / mins_in_year))


        df["power_without_plr_losses"] = power_without_plr_losses

        return df


def calculate_power_with_plr_losses(df, plr_value, col_name, initial_date):
        """
        Function for calculating the power values with PLR losses (%/year).

        "real version"
        power_with_plr_losses = (1 - plr_value)^years_from_initial_date * power_without_plr_losses

        "simplified version"
        power_with_plr_losses = (1 - plr_value * years_from_initial_data) * power_without_plr_losses

        1 minute is 1/(60*24*365) = 1/525600th of a year.
        """

        initial_date = pd.to_datetime(initial_date)

        mins_in_year = 60*24*365
        mins_in_day = 24*60
        power_without_plr = df[col_name]

        date_difference = df.index - initial_date
        mins_from_beginning = date_difference.days * mins_in_day + np.floor(date_difference.seconds / 60)

        power_with_plr_losses = (1 - np.abs(plr_value)/100 * (mins_from_beginning / mins_in_year)) * power_without_plr


        df[col_name] = power_with_plr_losses

        return df