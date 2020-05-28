import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data.d_utils import load_data
from src.SessionState import *
from src.recommend import run_recommmendation

DATA_DIR = './tests/test_streamlit_app_data'
CSV_FNAMES = {
    'val': os.path.join(DATA_DIR, 'tst-streamlit_val_users.csv'),
    'val-part-merged_sessions': os.path.join(DATA_DIR, 'tst-streamlit_val_users-part-merged_sessions.csv'),
    'test_ids': os.path.join(DATA_DIR, 'tst-streamlit_test_ids.txt'),
    'age_bkt': os.path.join(DATA_DIR, 'tst-streamlit_age_gender_bkts.csv'),
    'seasons': os.path.join(DATA_DIR, 'tst-streamlit_popular_seasons.csv')
}
MODEL = './models/finalized_LRmodel.sav'

class TestStreamLitApp:
    @pytest.fixture(scope='session')
    def dataset(self):
        return load_data(CSV_FNAMES, features=True)

    def test_recommend(self, dataset):
        expected_predictions = np.array(['NDF', 'US', 'other', 'FR', 'IT'], dtype='<U5')
        expected_ndcg = {'ndcg': 0.6309297535714575}
        predictions, ndcg = run_recommmendation(dataset, '87mebub9p4', MODEL)
        assert (expected_predictions == predictions).all()
        assert expected_ndcg == ndcg

        expected_predictions = ['No Destination Found', 'USA', 'Other',
                                'France', 'Italy']
        predictions, ndcg = run_recommmendation(dataset, '87mebub9p4', MODEL,
                                                full_name=True)
        assert expected_predictions == predictions
        assert expected_ndcg == ndcg


class TestSessionState():
    def test_SessionState(self):
        session_state = SessionState(user_name='', favorite_color='black')
        assert session_state.favorite_color == 'black'
    #
    # def test_get(self):
    #     session_state = get(user_name='Mary', favorite_color='black')
    #     assert 'Mary' == session_state.user_name
