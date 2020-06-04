# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:50:37 2020
dasdasfasfdasfdasad
@author: dguan
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_data():
    user = pd.read_csv('./airbnb-recruiting-new-user-bookings/train_users_2.csv')

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
    format = "%m/%d/%Y"
    try:
        user_date['First Booking Date'] = pd.to_datetime(user_date['First Booking Date'],
                                                         format=format)
    except:
        format = "%Y-%m-%d"
        user_date['First Booking Date'] = pd.to_datetime(user_date['First Booking Date'],
                                                         format=format)

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
    plt.xlabel('Booking Date', fontsize='xx-large')
    plt.title('Travelling Rate', fontsize='xx-large')
    plt.ylabel('Total (log)', fontsize='xx-large')
    plt.xticks(np.arange(1, 73, 6), ['Jan-10', 'July-10', 'Jan-11', 'July-11', 'Jan-12', 'July-12',
                                     'Jan-13', 'July-13', 'Jan-14', 'July-14', 'Jan-15'])
    st.pyplot()


def plot_country_most_visited(user_date):
    cou = ['US', 'Other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL', 'AU', 'PT']
    # cou=['CA']

    a_x = user_date[user_date['Country Destination'].isin(cou)]
    a_x = a_x.groupby(['Country Destination']).agg('count').reset_index()
    a_x = a_x[['Country Destination', 'id']].sort_values(['id'], ascending=False)
    a_x.plot.bar(x='Country Destination', y='id', rot=0, logy=True, legend=False, fontsize='xx-large')
    # plt.ylabel("Total",fontsize='xx-large')
    plt.xlabel('', fontsize='xx-large')
    plt.title('Travelling Rate by Country', fontsize='xx-large')
    st.pyplot()


def plot_age(user_date, cou):
    # cou=['US','Other','FR','CA','GB','ES','IT','DE','NL','AU','PT']
    # cou = ['AU', 'US']

    a_x = user_date[user_date['Country Destination'].isin(cou)]
    conditions = [
        (a_x['Age'] >= 18) & (a_x['Age'] <= 30),
        (a_x['Age'] >= 31) & (a_x['Age'] <= 40),
        (a_x['Age'] >= 41) & (a_x['Age'] <= 50),
        (a_x['Age'] >= 51) & (a_x['Age'] <= 60),
        (a_x['Age'] >= 61)]
    choices = ['18-30', '31-40', '41-50', '51-60', '60+']
    a_x['age'] = np.select(conditions, choices)
    a_x = a_x.groupby(['age', 'Gender']).agg('count').reset_index()
    a_x = a_x[['age', 'Gender', 'id']]
    a_x
    plt.figure(figsize=(10, 7))
    c = 0
    t = []
    for i in a_x['Gender'].unique():
        t.append([])
        for j in a_x['age'].unique():
            try:
                temp = a_x[(a_x['age'] == j) & (a_x['Gender'] == i)]['id'].iloc[0]
                t[c].append(temp)
            except:
                print('lol')
        c = c + 1

    bar = [1, 2, 3, 4, 5]
    barwidth = 0.8
    mal = plt.bar(bar, t[2], color='#2ECC71', width=barwidth, edgecolor='white')
    fem = plt.bar(bar, t[1], color='#BB8FCE', bottom=t[2], width=barwidth, edgecolor='white')
    unk = plt.bar(bar, t[0], color='#F4D03F', bottom=np.add(t[1], t[2]), width=barwidth, edgecolor='white')

    plt.ylabel("Total", fontsize='xx-large')
    plt.xlabel('Age Range', fontsize='xx-large')
    plt.title('Age Distribution with Selected Countries', fontsize='xx-large')
    plt.xticks(bar, choices, fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.legend((unk[0], fem[0], mal[0]), ('Unknown', 'Female', 'Male'), fontsize='xx-large')
    # plt.show()
    st.pyplot()


def visualization():
    # === Start Streamlit Application ===#
    st.title("Airbnb Data Analytics")
    st.image(Image.open('airbnb-recruiting-new-user-bookings/figs'
                        '/data_analytics_front_page.jpg'), use_column_width=True)
    st.markdown(
        """
            Interested in what are popular destination spots?\n\nWhat about who is 
            visiting them?\n\nAre there any patterns within the data?\n\nIf you 
            answered yes to any of these questions, please take a look at our data 
            analysis of Airbnb's country booking dataset to find out the answers and 
            other relevant information.
        """)

    user_date = load_data()

    # ===========part 1===========
    st.header("Part 1: Airbnb Travelers 101")
    st.markdown("This section focuses on investigating the lay of the land with "
                "Airbnb's country booking dataset i.e. exploring what countries are "
                "being visited, when is this happening, etc.")

    # ===========part 1.1 countries visited most===========
    st.subheader("Part 1.1 What countries are being visited the most?")

    plot_country_most_visited(user_date)

    # ===========part 1.2 when travelling takes place===========
    st.subheader("Part 1.2 When is the travelling taking place?")
    options = st.multiselect('Show travelling rate of the specific countries',
                             ['US', 'Other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL', 'AU', 'PT'], ['US', 'CA'])

    c_x = clean_data(user_date, options)

    plot_line_chart(c_x, options)

    # ===========part 1.3 who are the travels===========
    st.subheader("Part 1.3 Who are the travellers?")
    options1 = st.multiselect('Show age of the specific countries',
                             ['US', 'Other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL', 'AU', 'PT'], ['US', 'CA'])

    plot_age(user_date, options1)

    # plot_age_sub(user_date, options1)

    st.header("Part 2: Successful Booking vs No Booking")
    st.markdown("In ideal world for Airbnb, users, who visit their website, would end "
                "up making a booking. However, this isn't the case. We seek to "
                "investigate what are similarities and differences between users who "
                "make a successful booking vs users who do not book.")
    st.subheader("Part 2.1")
    st.subheader("Part 2.2")


if __name__ == "__main__":
    visualization()

