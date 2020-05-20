from collections import defaultdict
import glob
import os
from PIL import Image

import pandas as pd

class RecommendCountry():
    def __init__(self, seasons_csv=None):
        self.set_country_image()

        self.set_country_caption()

        if seasons_csv:
            self.seasons = pd.read_csv(seasons_csv, index_col='tfa_seasons')

    def set_country_image(self):
        countries = glob.glob('./airbnb-recruiting-new-user-bookings/figs/*')
        self.country_names = [os.path.basename(fname).split('.')[0] for fname in
                              countries]
        self.images = dict(zip(self.country_names, countries))

    def set_country_caption(self):
        self.captions = {}
        self.captions['IT'] = """
        Of the countries on our list, Italy is one of the most popular destination. 
        For an epic vacation, Italy is hard to beat. With Italian food and wine, 
        Renaissance art, gorgeous cities, ancient world history, and fabulous beach 
        towns, Italy does not disappoint!"""

        self.captions['ES'] ="""
        Sunny Spain is a great location for a mid-season holiday in Europe. The 
        cities of Barcelona and Madrid are of course must-dos. If possible, coincide 
        your visit with one of Spain’s many festivals"""

        self.captions['AU'] = """
        The land Down Under has so much diversity that you can find everything from 
        beaches to mountains, theme parks to ski resorts and vineyards to wildlife. 
        Visit the iconic Opera House in Sydney, go snorkelling at the Great Barrier 
        Reef and check out Kangaroo Island’s rugged beauty. You will truly be spoilt 
        for choice in Australia!"""

        self.captions['FR'] = """
        The list of reasons to visit France is endless! The world’s most visited 
        country is blessed with stunning landscape comprising of Alpine mountains, 
        beautiful meadows, farms, rivers and spectacular sea coast. France is the 
        leading light in our planet’s culture, performing arts and gastronomy."""

        self.captions['GB'] = """
        For such a small country, England packs a big punch. With jaw-droppingly 
        beautiful countryside, award-winning beaches and a whole lot of character, 
        England should top everyone’s must-visit bucket list."""

        self.captions['US'] = """
        The United States of America is a vast and incredibly diverse country. 
        Expect cosmopolitan megacities, quaint traditional villages and everything 
        in between. That’s not to mention this nation’s jaw-dropping natural beauty."""

        self.captions['DE'] = """
        One of the most beautiful countries in the world, Denmark is a small, 
        compact, and a very convenient country to visit. This amazing Nordic country 
        is known for its sturdy engineering, architecture, food, fashion, 
        art galleries & museums, castles & palaces, and truly world class cities"""

        #TODO give better captions for NDF and other
        self.captions['NDF'] = """
        Couldnt find you a country
        """

        self.captions['other'] = """
        Couldnt find you a country
        """

    def set_country_popular_age(self, csv_fname):
        """Set popular ages for country"""
        df = pd.read_csv(csv_fname)

        grouping = 'country_destination'
        genders = ['male', 'female']

        popular_age = defaultdict(dict)
        for cntry, cntry_df in df.groupby(grouping):
            gender_grpng = 'gender'

            for gndr, gndr_df in cntry_df.groupby(gender_grpng):
                age_bkt_idx = gndr_df['population_in_thousands'].argmax()
                popular_age[cntry][gndr] = gndr_df.iloc[age_bkt_idx]['age_bucket']

        self.popular_age = popular_age

    def get_popular_seasons(self, country):
        seasons_by_months = {
            'spring': ['March', 'April', 'May'],
            'summer': ['June', 'July', 'August'],
            'fall': ['September', 'October', 'November'],
            'winter': ['December', 'January', 'February']
        }
        idx = self.seasons[country].argmax()
        popular_season = self.seasons.iloc[idx].name
        str_months = ', '.join(seasons_by_months[popular_season])
        return popular_season, str_months

    def get_country_image(self, country):
        return Image.open(self.images[country])

    def get_image_caption(self, country):
        return self.captions[country]

    def get_country_info(self, country):
        img_fname = self.images[country.upper()]
