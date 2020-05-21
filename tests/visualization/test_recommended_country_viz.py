import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.visualization.recommended_countries_viz import RecommendCountry

TST_SEASONS = 'tests/test_streamlit_app_data/tst-streamlit_popular_seasons.csv'

TST_AGE1 = 'tests/visualization/test_age_gender_bkts.csv'
TST_AGE2 = 'tests/test_streamlit_app_data/tst-streamlit_age_gender_bkts.csv'

class TestCountryRecommender():
    @pytest.fixture(scope='session')
    def country_rec(self):
        return RecommendCountry(seasons_csv=TST_SEASONS)

    def test_set_country_popular_age(self, country_rec):
        expected_output = {'AU': {'male': '25-29', 'female': '30-34'}}
        country_rec.set_country_popular_age(csv_fname=TST_AGE1)
        assert expected_output == country_rec.popular_age

        country_rec.set_country_popular_age(csv_fname=TST_AGE2)
        assert country_rec.popular_age['AU']['female'] == '30-34'
        assert country_rec.popular_age['US']['female'] == '50-54'
        assert country_rec.popular_age['US']['male'] == '20-24'

    def test_get_popular_seasons(self, country_rec):
        country = 'US'
        expected_season_and_months = ('spring', 'March, April, May')
        assert expected_season_and_months == country_rec.get_popular_seasons(country)
