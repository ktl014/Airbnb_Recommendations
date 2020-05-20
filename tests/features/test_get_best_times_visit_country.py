import os
import shutil
import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.features.get_best_times_visit_country import *

class TestCountryTimeFeatures():
    @pytest.fixture(scope='session')
    def times(self):
        return [
             20090319043255,
             20090523174809,
             20090609231247,
             20091031060129,
             20091208061105
        ]

    @pytest.fixture(scope='session')
    def countries(self):
        return [
            'NDF', 'NDF', 'US', 'other', 'US'
        ]

    def test_get_seasons(self, times):
        expected_output = ['spring', 'spring', 'summer', 'fall', 'winter']
        seasons = get_seasons(pd.DataFrame(times, columns=['timestamp_first_active']))
        assert expected_output == seasons['tfa_seasons'].to_list()

    def test_generate_season_features(self, times, countries):
        users = pd.DataFrame({'timestamp_first_active': times,
                              'country_destination': countries})
        expected_output_fname = './tests/features/popular_seasons.csv'
        generate_seasons_features(users, expected_output_fname)
        assert os.path.exists(expected_output_fname)
        data = pd.read_csv(expected_output_fname, index_col='tfa_seasons')
        print(data)
        assert data.index.name == 'tfa_seasons'
        assert data['US'].iloc[2] == 1.0
        assert data['US'].iloc[3] == 1.0

        if os.path.exists(expected_output_fname):
            os.remove(expected_output_fname)
