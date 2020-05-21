import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.data.d_utils import sample_data, preprocess_data, load_data
from src.SessionState import *

DATA_DIR = './airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'val': os.path.join(DATA_DIR, 'val_users.csv'),
    'val-part-merged_sessions': os.path.join(DATA_DIR, 'val_users-part-merged_sessions.csv'),
    'test_ids': os.path.join(DATA_DIR, 'test_ids.txt'),
    'age_bkt': os.path.join(DATA_DIR, 'age_gender_bkts.csv'),
    'seasons': os.path.join(DATA_DIR, 'popular_seasons.csv')
}

class TestStreamLitApp:
    @pytest.fixture(scope='session')
    def test_data(self):
        with open('./tests/streamlit_test_case.json', 'r') as fp:
            data_dict = json.load(fp)

        # convert dictionaries into dataframes
        data = {
            key: pd.DataFrame(data_dict[key])
            for key in data_dict
        }
        return data

    def test_load_data(self):
        datasets = load_data(CSV_FNAMES)
        assert hasattr(datasets, 'users')
        assert isinstance(datasets.users, pd.DataFrame)
        datasets = load_data(CSV_FNAMES, features=True)
        assert hasattr(datasets, 'users_feat')
        assert isinstance(datasets.users_feat, pd.DataFrame)


    def test_sample_data(self, test_data):
        expected_id = input = '0nb7ohdmvk'
        sample, id = sample_data(test_data['raw'], input)

        assert id == expected_id
        assert sample.shape == (1,16)

        test_ids = ['rx82tiu9oy', '6i683pijvk']
        sample, id = sample_data(test_data['raw'], test_ids=test_ids)
        assert isinstance(sample, pd.DataFrame)
        assert id in test_ids

    def test_preprocess_data(self, test_data):
        x, y = preprocess_data(test_data['features'])
        assert 'country_destination' not in x.columns
        assert 'id' not in x.columns
        assert (y == pd.Series(['NDF', 'NDF'], name='country_destination')).all()

class TestSessionState():
    def test_SessionState(self):
        session_state = SessionState(user_name='', favorite_color='black')
        assert session_state.favorite_color == 'black'
    #
    # def test_get(self):
    #     session_state = get(user_name='Mary', favorite_color='black')
    #     assert 'Mary' == session_state.user_name
