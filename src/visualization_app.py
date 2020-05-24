# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:50:37 2020

@author: dguan
"""
from datetime import datetime
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    user = pd.read_csv('airbnb-recruiting-new-user-bookings/train_users_2.csv')

    # clean up data
    user = user[(user['age'] < 110) & (user['age'] > 0) & (user['date_first_booking'] != '0')]
    user = user.drop(['signup_flow', 'first_affiliate_tracked'], axis=1)
    user = user.rename(columns={'date_account_created': 'Date of Account Created',
                                'timestamp_first_active': 'Activity Time',
                                'date_first_booking': 'First Booking Date',
                                'gender': 'Gender', 'age': 'Age', 'signup_method': 'Sign Up Method',
                                'language': 'Language', 'affiliate_channel': 'Affiliate Channel',
                                'affiliate_provider': 'Affiliate Provider', 'signup_app': 'Signup Place',
                                'first_device_type': 'Device Type', 'first_browser': 'Browser',
                                'country_destination': 'Country Destination'})
    user_date = user[(user['First Booking Date'] != '0') & (user['Age'] != 0)]
    user_date['Age'] = user_date['Age'].astype(int)
    user_date = user_date[(user_date['Age'] >= 18) & (100 >= user_date['Age'])]
    user_date.dropna(inplace=True)
    user_date['First Booking Date'] = pd.to_datetime(user_date['First Booking Date'], format="%m/%d/%Y")
    user_date['First Booking Date'] = user_date['First Booking Date'].apply(lambda y: y.strftime('%Y-%m'))
    user_date['Gender'].replace({0: 'Unknown', 1: 'Male', 2: 'Female'}, inplace=True)

    return user_date


def clean_data(x, cou):
    x = x[x['Country Destination'].isin(cou)]
    c_x = x.sort_values(['First Booking Date']).groupby(['First Booking Date', 'Country Destination']).agg(
        'count').reset_index()

    c_x = c_x[['First Booking Date', 'Country Destination', 'id']]

    if cou:
        temp = list(c_x[c_x['Country Destination'] == cou[0]]['First Booking Date'])

    for i in cou:
        a = set(temp).difference(list(c_x[c_x['Country Destination'] == i]['First Booking Date']))
        for j in a:
            c_x.loc[len(c_x)] = [j, i, 1]
        a = set(list(c_x[c_x['Country Destination'] == i]['First Booking Date'])).difference(temp)
        for j in a:
            c_x.loc[len(c_x)] = [j, cou[0], 1]

    c_x.sort_values(by=['First Booking Date'], inplace=True)
    c_x.reset_index(drop=True, inplace=True)

    return c_x


def plot_line_chart(c_x, cou):

    plt.figure(figsize=(10, 5))

    for i in cou:
        if cou is None:
            continue
        data = c_x[c_x['Country Destination'] == i]
        data = data.append(data)
        plt.plot('First Booking Date', 'id', data=c_x[c_x['Country Destination'] == i])
        plt.yscale('log')

    plt.legend(tuple(cou))
    plt.xlabel('Booking Date', fontsize=14)
    plt.title('Travelling Rate', fontsize=20)
    plt.ylabel('Total (log)', fontsize=14)
    plt.xticks(np.arange(1, 73, 6), ['Jan-10', 'July-10', 'Jan-11', 'July-11', 'Jan-12', 'July-12',
                                     'Jan-13', 'July-13', 'Jan-14', 'July-14', 'Jan-15'])
    st.pyplot()


# === Start Streamlit Application ===#
st.title("Airbnb System Visualization")
st.markdown(
    """
        This is a demo of a Streamlit app that visualize the Airbnb dataset.
        [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
    """)

user_date = load_data()

options = st.multiselect('Show travelling rate of the specific countries',
                         ['US', 'Other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL', 'AU', 'PT'], ['US', 'CA'])

c_x = clean_data(user_date, options)

plot_line_chart(c_x, options)
