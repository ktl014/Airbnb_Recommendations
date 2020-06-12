import json
import os
import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.data.d_utils import *

TEST_CSV1 = './tests/test-train_users_2-tst.csv'

DATA_DIR = './tests/test_streamlit_app_data'
TEST_CSV_DATASETS = {
    'val': os.path.join(DATA_DIR, 'tst-streamlit_val_users.csv'),
    'val-part-merged_sessions': os.path.join(DATA_DIR, 'tst-streamlit_val_users-part-merged_sessions.csv'),
    'test_ids': os.path.join(DATA_DIR, 'tst-streamlit_test_ids.txt'),
    'age_bkt': os.path.join(DATA_DIR, 'tst-streamlit_age_gender_bkts.csv'),
    'seasons': os.path.join(DATA_DIR, 'tst-streamlit_popular_seasons.csv')
}


class TestDUtils():
    @pytest.fixture(scope='session')
    def test_dataset(self):
        return pd.read_csv(TEST_CSV1)

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

    def test_read_in_dataset(self, test_dataset):
        df = read_in_dataset(TEST_CSV1, keep_id=True)
        assert test_dataset.equals(df)

        dataset = test_dataset.drop('id', axis=1)
        df = read_in_dataset(TEST_CSV1, verbose=True)
        assert dataset.equals(df)

    def test_load_data(self):
        datasets = load_data(TEST_CSV_DATASETS)
        assert hasattr(datasets, 'users')
        assert isinstance(datasets.users, pd.DataFrame)
        datasets = load_data(TEST_CSV_DATASETS, features=True)
        print(datasets)
        assert hasattr(datasets, 'users_feat')
        assert isinstance(datasets.users_feat, pd.DataFrame)

    def test_sample_data(self, test_data):
        # ids = ['87mebub9p4', 'k6np330cm1']
        expected_id = input = '0nb7ohdmvk'
        sample, id = sample_data(test_data['raw'], input)

        assert id == expected_id
        assert sample.shape == (1, 16)

        test_ids = ['rx82tiu9oy', '6i683pijvk']
        sample, id = sample_data(test_data['raw'], test_ids=test_ids)
        assert isinstance(sample, pd.DataFrame)
        assert id in test_ids

    def test_preprocess_data(self, test_data):
        x, y = preprocess_data(test_data['features'])
        assert 'country_destination' not in x.columns
        assert 'id' not in x.columns
        assert (y == pd.Series(['NDF', 'NDF'], name='country_destination')).all()


class TestExpFeatures():
    @pytest.fixture(scope='session')
    def processed_dataset(self):
        test_examples = pd.read_json('./tests/LRmodel-v001_test_case.json')
        return test_examples

    def test_get_features(self, processed_dataset):
        expected_features = processed_dataset.columns
        exp = ExpFeatures(expected_features, stats=True, ratios=True, casted=True)
        assert len(expected_features) == len(exp.get_features(verbose=True))

    def test_experiment_features(self, processed_dataset):
        expected_n_features = 165
        output = experiment_features(processed_dataset)
        assert output.shape[1] == expected_n_features

        output = experiment_features(processed_dataset, stats=True, ratios=True,
                                     casted=True, verbose=True)
        expected_n_features = 1259
        assert output.shape[1] == expected_n_features





