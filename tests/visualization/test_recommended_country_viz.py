import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.visualization.recommended_countries_viz import RecommendCountry

TST_CSV = 'tests/visualization/test_age_gender_bkts.csv'

class TestCountryRecommender():
    @pytest.fixture(scope='session')
    def country_rec(self):
        return RecommendCountry()

    def test_set_country_popular_age(self, country_rec):
        expected_output = {'AU': {'male': '25-29', 'female': '30-34'}}
        country_rec.set_country_popular_age(csv_fname=TST_CSV)

        assert expected_output == country_rec.popular_age
