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

@st.cache(suppress_st_warning=True)
def load_data():
    """
    load data from train_user_2.csv for future use in part 1

    """
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
    """
        Clean dataframe by removing NA values and keep useful columns.

        x: dataframe
        cou: list of countries

     """

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


def plot_country_time_series(c_x, cou):
    """
    Show plot of visited frquency by date with selected countries.

    Parameters
    ----------
    c_x: dataframe
    cou: list of countries

    """
    plt.figure(figsize=(11, 9))

    for i in cou:
        if cou is None:
            continue
        data = c_x[c_x['Country Destination'] == i]
        data = data.append(data)
        plt.plot('First Booking Date', 'id', data=c_x[c_x['Country Destination'] == i])
        plt.yscale('log')
    fontsize = 'xx-large'

    plt.legend(tuple(cou), fontsize=fontsize)
    plt.xlabel('Bookings from 2010-2015', fontsize=fontsize)
    plt.title('Country Booking Trends', fontsize=fontsize)
    plt.ylabel('Number of Bookings (log)', fontsize=fontsize)
    plt.xticks(np.arange(1, 73, 6), ['Jan-10', 'July-10', 'Jan-11', 'July-11', 'Jan-12', 'July-12',
                                     'Jan-13', 'July-13', 'Jan-14', 'July-14',
                                     'Jan-15'], fontsize=fontsize, rotation=30)
    plt.yticks(fontsize=fontsize)
    st.pyplot()


def plot_country_most_visited(user_date):

    """
    Show visualization of visited frquency with selected countries.

    Parameters
    ----------
    user_date: dataframe

    """
    cou = ['US', 'Other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL', 'AU', 'PT']
    # cou=['CA']
    plt.figure(figsize=(10, 5))
    fontsize = 'large'
    a_x = user_date[user_date['Country Destination'].isin(cou)]
    a_x = a_x.groupby(['Country Destination']).agg('count').reset_index()
    a_x = a_x[['Country Destination', 'id']].sort_values(['id'], ascending=False)
    a_x.plot.bar(x='Country Destination', y='id', rot=0, logy=True, legend=False,
                 fontsize=fontsize)
    plt.ylabel("Total Bookings (logged)",fontsize=fontsize)
    plt.xlabel('Countries', fontsize=fontsize)
    plt.title('Frequency of Country Bookings from 2010-2015', fontsize=fontsize)
    st.pyplot()


def plot_age(user_date, cou):
    """
    Show visualization of visitor demograph with selected countries.

    Parameters
    ----------
    user_date: dataframe
    cou: list of countries

    """
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

    fontsize = 'xx-large'
    plt.ylabel("Total", fontsize=fontsize)
    plt.xlabel('Age Range', fontsize=fontsize)
    plt.title('Age Distribution with Selected Countries', fontsize=fontsize)
    plt.xticks(bar, choices, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend((unk[0], fem[0], mal[0]), ('Unknown', 'Female', 'Male'), fontsize=fontsize)
    st.pyplot()

@st.cache(suppress_st_warning=True)
def load_data_part2():
    """"
    load data for part 2 visualization from session.csv and train_user_2.csv

    prepare for further use of making plots in part 2

    Returns:
        DataFrame df,
            the session information of users whose desination is known,
            (the intersection between session.csv and train_user_2.csv)

        dict device_dic,
            {user_id: device_type}, the device type of each user

        dict result_dic,
            {user_id: country_destination}, the destination of each user

        dict lang_dic,
            {user_id: language}, the language of each user

    """

    s_df = pd.read_csv('./airbnb-recruiting-new-user-bookings/sessions.csv')
    t_df = pd.read_csv('./airbnb-recruiting-new-user-bookings/train_users_2.csv')

    result_dic = t_df.set_index('id')['country_destination'].to_dict()
    lang_dic = t_df.set_index('id')['language'].to_dict()
    device_dic = t_df.set_index('id')['first_device_type'].to_dict()

    s_df['country_destination'] = s_df.user_id.apply(lambda x: result_dic[x] if x in result_dic.keys() else '0')
    s_df['lang'] = s_df.user_id.apply(lambda x: lang_dic[x] if x in lang_dic.keys() else '0')

    df = s_df[s_df['country_destination'] != '0']

    return df, device_dic, result_dic, lang_dic


def plot_avg_time_action_type(df, device_dic, result_dic):
    """
    plot for part 2.1 what steps are taken for booking a travel destination

    Args:
        df (dataframe): the input dataframe that we are going to plot about

        device_dic (dict): {user_id: device_type}, the device type of each user

        result_dic (dict): {user_id: country_destination}, the destination of each user

    """
    tmpdf = pd.DataFrame(df.groupby(['user_id', 'action_type'])['secs_elapsed'].agg(np.sum))
    tmpdf['id'] = tmpdf.index.map(lambda x: x[0])
    tmpdf['action'] = tmpdf.index.map(lambda x: x[1])
    tmpdf['device_type'] = tmpdf.id.apply(lambda x: device_dic[x] if x in device_dic.keys() else '0')
    tmpdf['country_destination'] = tmpdf.id.apply(lambda x: result_dic[x] if x in result_dic.keys() else '0')

    str1 = 'Unsuccessful booking'
    str2 = 'Successful booking'

    fontsize = 'large'
    ax = pd.DataFrame(
        {str2: tmpdf[tmpdf['country_destination'] != 'NDF'].groupby('action_type')['secs_elapsed'].agg(np.mean),
         str1: tmpdf[tmpdf['country_destination'] == 'NDF'].groupby('action_type')['secs_elapsed'].agg(np.mean)
         }).plot.barh(figsize=(12,5))

    plt.title('Average Time spent on an Action', fontsize=fontsize)
    plt.ylabel('Action type', fontsize=fontsize)
    plt.xlabel('Time/ms', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    st.pyplot()


def plot_language_time(df, device_dic, result_dic, lang_dic):
    """"
    make a plots for part 2.2 and part 2.3.

    2.2. average time that a single user spends during booking with respect to languages

    2.3. devices have higher success rates when booking a destination

    Args:
        df (dataframe): the input dataframe that we are going to plot about

        device_dic (dict): {user_id: device_type}, the device type of each user

        result_dic (dict): {user_id: country_destination}, the destination of each user

        lang_dic (dict): {user_id: language}, the language of each user


    """

    id_grouped = df.groupby('user_id')

    id_df = pd.DataFrame(id_grouped['secs_elapsed'].agg([np.sum, np.mean, np.std]))
    id_df['country_destination'] = id_df.index.map(lambda x: result_dic[x] if x in result_dic.keys() else '0')
    id_df['language'] = id_df.index.map(lambda x: lang_dic[x] if x in lang_dic.keys() else '0')
    id_df['device'] = id_df.index.map(lambda x: device_dic[x] if x in device_dic.keys() else '0')
    full_lang = {'en': 'English', 'zh': 'Chinese', 'ko': 'Korean', 'fr': 'French', 'es': 'Spanish',
                 'de': 'German', 'ru': 'Russian', 'it': 'Italian', 'ja': 'Japanese', 'pt': 'Portuguese'}
    id_df['full_language'] = id_df.language.apply(lambda x: full_lang[x] if x in full_lang.keys() else '0')

    id_df[id_df['full_language'] != '0'].boxplot(column='sum', by='full_language', showfliers=False, patch_artist=True)
    plt.title('Average time that a single user spend during booking')
    plt.ylabel('Time/ms')
    plt.xlabel('Language')
    plt.xticks(rotation="45")
    st.pyplot()


    # ===========part 2.3 ============
    st.subheader("Part 2.3.")
    st.markdown("Which devices have higher success rates when booking a destination?")

    st.subheader("Part 2.3 Do devices play a role in this comparisons?")

    device_df = pd.DataFrame({'NDF': id_df[id_df['country_destination'] == 'NDF'].groupby('device')['sum'].agg(np.size),
                              'DF': id_df[id_df['country_destination'] != 'NDF'].groupby('device')['sum'].agg(np.size)
                              })

    str1 = 'Unsuccessful booking'
    str2 = 'Successful booking'

    device_usernum = dict(id_df.groupby('device')['country_destination'].agg(np.size))
    device_df['total_number'] = device_df.index.map(lambda x: device_usernum[x] if x in device_usernum.keys() else '0')

    device_df[str1] = device_df.eval('NDF / total_number * 100')
    device_df[str2] = device_df.eval('DF / total_number * 100')
    device_df = device_df.sort_values(by="total_number", ascending=True)

    device_df[[str2, str1]].plot.barh(stacked=True,legend=False,figsize=(12,7))
    plt.ylabel('Device type(sorted)')
    plt.xlabel('Percentage',fontsize="xx-large")
    plt.title('Success Booking Rate of each Device')
    plt.legend(loc='lower left',bbox_to_anchor=(-0.1,-0.1))
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
    df, device_dic, result_dic, lang_dic = load_data_part2()

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
    options = st.multiselect('Select countries to view their travelling rate',
                             ['US', 'Other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL', 'AU', 'PT'], ['US', 'CA'])

    c_x = clean_data(user_date, options)

    plot_country_time_series(c_x, options)

    # ===========part 1.3 who are the travellerss=========== #
    st.subheader("Part 1.3 Who are the travellers?")
    options1 = st.multiselect('Select countries to see their age distribution',
                              ['US', 'Other', 'FR', 'CA', 'GB', 'ES', 'IT', 'DE', 'NL', 'AU', 'PT'], ['US', 'CA'])

    plot_age(user_date, options1)

    # plot_age_sub(user_date, options1)

    st.header("Part 2: Successful Booking vs No Booking")
    st.markdown("In ideal world for Airbnb, users, who visit their website, would end "
                "up making a booking. However, this isn't the case. We seek to "
                "investigate what are similarities and differences between users who "
                "make a successful booking vs users who do not book.")


    df, device_dic, result_dic, lang_dic = load_data_part2()

    # ===========part 2.1===========
    st.subheader("Part 2.1")
    st.markdown("What steps are taken for booking a travel destination?")
    plot_avg_time_action_type(df, device_dic, result_dic)

    # ===========part 2.2 & 2.3===========
    st.subheader("Part 2.2")
    st.markdown("Is there any difference in time consuming when using different languages?")

    st.subheader("Part 2.1 Is there a pattern in the elapsed time of an action?")
    plot_avg_time_action_type(df, device_dic, result_dic)

    # ===========part 2.2 and 2.3 =========== #
    st.subheader("Part 2.2 Delving deeper into the elapsed time")
    st.markdown("Is there a difference in the elapsed time when using an international interface?")

    plot_language_time(df, device_dic, result_dic, lang_dic)


if __name__ == "__main__":
    visualization()
