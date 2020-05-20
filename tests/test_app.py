import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.app import sample_data, preprocess_data, load_data

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
        datasets = load_data()
        assert hasattr(datasets, 'users')
        assert isinstance(datasets.users, pd.DataFrame)
        datasets = load_data(features=True)
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