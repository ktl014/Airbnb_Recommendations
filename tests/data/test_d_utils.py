import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

test_csv = './airbnb-recruiting-new-user-bookings/train_users_2.csv'

from src.data.d_utils import *

class TestDUtils():
    @pytest.fixture(scope='session')
    def dataset(self):
        return pd.read_csv(test_csv)

    def test_read_in_dataset(self, dataset):
        df = read_in_dataset(test_csv, keep_id=True)
        assert dataset.equals(df)

        dataset = dataset.drop('id', axis=1)
        df = read_in_dataset(test_csv, verbose=True)
        assert dataset.equals(df)

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





