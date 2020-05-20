# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:50:37 2020

@author: dguan
"""
from collections import namedtuple
import os
import pickle
import random
import sys
from pathlib import Path
# PROJECT_DIR = Path(__file__).resolve()
# print(str(PROJECT_DIR.parents[1]))
# sys.path.insert(0, PROJECT_DIR.parents[1])

import streamlit as st
from sklearn.preprocessing import LabelEncoder

from data.d_utils import read_in_dataset
from eval_model import predict, evaluate_model
import SessionState

MODEL = './models/finalized_LRmodel.sav'

DATA_DIR = './airbnb-recruiting-new-user-bookings'

CSV_FNAMES = {
    'val': os.path.join(DATA_DIR, 'val_users.csv'),
    'val-part-merged_sessions': os.path.join(DATA_DIR, 'val_users-part-merged_sessions.csv'),
    'test_ids': os.path.join(DATA_DIR, 'test_ids.txt')
}

# load dataset
@st.cache
def load_data(features=False):
    DATASETS = {}

    DATASETS['users'] = read_in_dataset(CSV_FNAMES['val'], keep_id=True)

    if features:
        dataset_type = 'part-merged_sessions'
        DATASETS['users_feat'] = read_in_dataset(CSV_FNAMES[f'val-{dataset_type}'],
                                                 keep_id=True)

    Airbnb = namedtuple('Airbnb', list(DATASETS.keys()))
    return Airbnb(**DATASETS)

@st.cache
def load_model(fname):
    return pickle.load(open(fname, 'rb'))

def sample_data(data, id=None, test_ids=None):
    if test_ids:
        id = random.choice(test_ids)
    sample = data[data['id'] == id].reset_index(drop=True)
    return sample, id

def preprocess_data(data):
    label = data.pop('country_destination')
    data = data.drop('id', axis=1)
    return data, label

def run():

    data, predictions, id = None, [] ,0
    session_state = SessionState.get(id=id, data=data, prediction=predictions)

    # Load dataset
    dataset = load_data(features=True)
    test_ids = open(CSV_FNAMES['test_ids'], 'r').read().splitlines()

    # Get user id
    st.header('Generate User')
    if st.button('Click here to Generate User ID'):

        session_state.X, session_state.id = sample_data(dataset.users, test_ids=test_ids)
        st.write(session_state.id)

    st.subheader('Raw Data')
    if st.checkbox('Show data') and session_state.id:
        FEAT_TO_DISPLAY = ['id', 'country_destination',
                           'date_account_created', 'gender', 'age',
                           'signup_method', 'signup_app', 'language']
        show_data = session_state.X[FEAT_TO_DISPLAY]
        st.dataframe(show_data)

    # Recommend based off id
    st.header('Recomended Countries')
    if st.button('Click here to Recommend Countries'):
        le = LabelEncoder().fit(dataset.users['country_destination'])

        d, session_state.id = sample_data(dataset.users_feat, id=session_state.id)
        X, y = preprocess_data(d)

        # Load model
        model = load_model(MODEL)

        # Make model inference
        # then postprocess the prediction
        predictions = predict(model, X)
        predictions = le.inverse_transform(predictions[0])
        st.write(predictions)
        st.write(session_state.id)

        #TODO Display Model Predictions
    else:
        st.write("Press the above button..")

if __name__ == '__main__':
    st.title("Airbnb Recomendation System")
    st.markdown(
        """
            This is a demo of a Streamlit app that shows Airbnb recomendation for travellers. 
            [See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
        """)
    run()
