"""Get best times to visit a country feature

Module for `get_seasons` and generating the seasons feature

"""
import numpy as np
import pandas as pd

def get_seasons(df, time_col='timestamp_first_active', datefmt='%Y%m%d%H%M%S'):
    """ Get seasons given a dataframe and returns it within the dataset

    Args:
        df (pd.DataFrame): Dataset
        time_col (str): Time column to process season mapping for
        datefmt (str): Datetime format for the time column

    Returns:
        pd.DataFrame: Dataset with season feature as `tfa_seasons`

    """
    seasons = ['winter', 'winter', 'spring',
               'spring', 'spring', 'summer', 'summer',
               'summer', 'fall', 'fall', 'fall', 'winter']
    month_to_season = dict(zip(range(1, 13), seasons))
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], format=datefmt)
    df['tfa_seasons'] = df[time_col].dt.month.map(month_to_season)
    return df

def generate_seasons_features(users, output_fname):
    """ Generate seasons dataset into a pivot table and saves it as a csv file

    Used primarily for recommendation dashboard.

    Args:
        users (pd.DataFrame): Users dataset
        output_fname (str): Output filename

    Returns:

    """
    users = users[['timestamp_first_active', 'country_destination']]

    users = get_seasons(users)

    data = pd.pivot_table(users,
                          columns=['country_destination'],
                          index=['tfa_seasons'],
                   aggfunc='count')
    data.columns = data.columns.droplevel()

    data.to_csv(output_fname)
    return data
#
# generate_seasons_features(pd.read_csv('airbnb-recruiting-new-user-bookings/train_users_2.csv'),
#                           './airbnb-recruiting-new-user-bookings/popular_seasons.csv')