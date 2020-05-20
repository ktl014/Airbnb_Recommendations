import numpy as np
import pandas as pd

def get_seasons(df, time_col='timestamp_first_active', datefmt='%Y%m%d%H%M%S'):
    seasons = ['winter', 'winter', 'spring',
               'spring', 'spring', 'summer', 'summer',
               'summer', 'fall', 'fall', 'fall', 'winter']
    month_to_season = dict(zip(range(1, 13), seasons))
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col], format=datefmt)
    df['tfa_seasons'] = df[time_col].dt.month.map(month_to_season)
    return df

def generate_seasons_features(users, output_fname):
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