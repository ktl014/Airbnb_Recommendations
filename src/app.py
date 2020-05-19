# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:50:37 2020

@author: dguan
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from random import randrange

DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

st.title("Airbnb Recomendation System")
st.markdown(
    """
        This is a demo of a Streamlit app that shows Airbnb recomendation for travellers. 
        [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
        """)

# load the data
data = pd.read_csv('./airbnb-recruiting-new-user-bookings/new_train_users_2.csv')
rows_data = data.shape[0]

# generate user id
st.header('Generate User')
if st.button('Click here to Generate User ID'):
    rnd = randrange(rows_data)
    user_id = list(data[data.index == rnd]['id'])[0]
    st.write(user_id)
else:
    st.write('click the button to generate user ID')

st.header('Raw Data')
check_data = st.checkbox('Show Dataframe')
if check_data:
    show_data = data[data.index == 1][['id', 'gender', 'age', 'country_destination']]
    st.dataframe(show_data)
# st.dataframe(pd.DataFrame({
#     # 'feature 1': [1, 2, 3, 4],
#     # 'feature 2': [10, 20, 30, 40],
#     # 'feature 3': [100, 20, 30, 40],
#     # 'feature 4': [10, 200, 30, 40]
# }))

# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])
#
# st.line_chart(chart_data)

st.header('Recomended Countries')
st.button('Click here to Recommend Countries')
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
st.map(map_data)

# @st.cache(persist=True)
# def load_data(nrows):
#         data = pd.read_csv(DATA_URL, nrows=nrows)
#         lowercase = lambda x: str(x).lower()
#         data.rename(lowercase, axis="columns", inplace=True)
#         data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
#         return data


# data = load_data(100000)

# hour = st.slider("Hour to look at", 0, 23)

# data = data[data[DATE_TIME].dt.hour == hour]

# st.subheader("Geo data between %i:00 and %i:00" % (hour, (hour + 1) % 24))
# midpoint = (np.average(data["lat"]), np.average(data["lon"]))

# st.write(pdk.Deck(
#     map_style="mapbox://styles/mapbox/light-v9",
#     initial_view_state={
#         "latitude": midpoint[0],
#         "longitude": midpoint[1],
#         "zoom": 11,
#         "pitch": 50,
#     },
#     layers=[
#         pdk.Layer(
#             "HexagonLayer",
#             data=data,
#             get_position=["lon", "lat"],
#             radius=100,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             pickable=True,
#             extruded=True,
#         ),
#     ],
# ))

# st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
# filtered = data[
#     (data[DATE_TIME].dt.hour >= hour) & (data[DATE_TIME].dt.hour < (hour + 1))
# ]
# hist = np.histogram(filtered[DATE_TIME].dt.minute, bins=60, range=(0, 60))[0]
# chart_data = pd.DataFrame({"minute": range(60), "pickups": hist})

# st.altair_chart(alt.Chart(chart_data)
#     .mark_area(
#         interpolate='step-after',
#     ).encode(
#         x=alt.X("minute:Q", scale=alt.Scale(nice=False)),
#         y=alt.Y("pickups:Q"),
#         tooltip=['minute', 'pickups']
#     ), use_container_width=True)

# if st.checkbox("Show raw data", False):
#         st.subheader("Raw data by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
#         st.write(data)
