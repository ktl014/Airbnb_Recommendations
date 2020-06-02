""" Data Utilities

Data utilities module currently contains helper functions for the following:
- reading datasets
- loading & processing datasets for the recommendation dashboard
- feature engineering ablation study

"""
from collections import namedtuple
import os
import random

import pandas as pd
from tqdm import tqdm

def read_in_dataset(csv_fname, raw=False, chunksize = 50000, keep_id=False, verbose=False):
    """ Read in one of the Salary datasets

    Args:
        csv_fname (str): Abs path of the dataset (e.g. test_features.csv, train_features.csv)
        raw (bool): Flag for raw or processed data. Default is raw
        verbose (bool): Print out verbosity

    Returns:
        pd.DataFrame: dataset

    """
    df_list = []
    for df_chunk in tqdm(pd.read_csv(csv_fname, chunksize=chunksize)):

        # Can process each chunk of dataframe here
        # clean_data(), feature_engineer(),fit()

        df_list.append(df_chunk)
    # Merge all dataframes into one dataframe
    df = pd.concat(df_list)
    # Delete the dataframe list to release memory
    del df_list

    if not keep_id and 'id' in df.columns:
        df = df.drop('id', axis=1)

    if verbose:
        print('\n{0:*^80}'.format(' Reading in the dataset '))
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.info())
        print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
        print(df.head())
    return df

def load_data(CSV_FNAMES, features=False):
    """Load recommendation dataset into tuple collection object, Airbnb

    Dataset will contain at default the raw dataset. Optionally, the feature
    engineered dataset can be included by turning on the `features` argument.

    Args:
        CSV_FNAMES (dict): Dictionary containing csv absolute paths
        features (bool): Flag for including processed dataset. Default is False.

    Returns:
        Airbnb: Airbnb dataset

    """
    datasets = {}

    datasets['users'] = read_in_dataset(CSV_FNAMES['val'], keep_id=True)

    if features:
        dataset_type = 'part-merged_sessions'
        datasets['users_feat'] = read_in_dataset(CSV_FNAMES[f'val-{dataset_type}'],
                                                 keep_id=True)

    Airbnb = namedtuple('Airbnb', list(datasets.keys()))
    return Airbnb(**datasets)

def sample_data(data, id=None, test_ids=None):
    """ Sample dataset for a single instance ~ used for recommendation dashboard

    Args:
        data (pd.DataFrame): Dataset
        id (str): User ID. Default is None.
        test_ids (list): List of test ids to sample from. Default is None

    Returns:
        pd.DataFrame: Sampled dataset
        str: User ID of the sampled dataset

    """
    if test_ids:
        id = random.choice(test_ids)
    sample = data[data['id'] == id].reset_index(drop=True)
    return sample, id

def preprocess_data(data):
    """ Preprocess dataset ~ used for recommendation dashboard

    Preprocessing is done here by popping the label and removing the id. Function
    largely assumes that the dataset is already processed and only needs to remove the
    label and id.

    Args:
        data (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset
        str: Label


    """
    label = data.pop('country_destination')
    data = data.drop('id', axis=1)
    return data, label

class ExpFeatures():
    """ ExperimentalFeature object for running feature engineered ablation studies

    Part of project was to investigate how model performance is influenced by certain
    feature engineerings, so ExpFeatures() is used to streamline these studies.

    """
    def __init__(self, features_list, stats=False, ratios=False, casted=False):
        """Initializes ExpFeatures

        Args:
            features_list (list): List of features
            stats (bool): statsTime Flag. Default is False
            ratios (bool): ratiosTime Flag. Default is False
            casted (bool): castedSessions Flag. Default is False
        """
        # Grab all feature engineerings associated with the given flags
        self.st_ = ['n_actions_per_user', 'n_distinct_action_detail', 'n_distinct_action_types',
               'n_distinct_actions', 'n_distinct_device_types']

        self.statistical_features = [i for i in features_list if i.endswith('_elapsed')]  + self.st_

        self.ratios = ['ratio_distinct_actions', 'ratio_distinct_actions_types',
                       'ratio_distinct_action_details', 'ratio_distinct_devices']

        self.casted_features = [i for i in features_list if i.endswith('_ratio')]

        # baseline features
        self.features = list(set(features_list).difference(
            set(self.statistical_features + self.casted_features + self.ratios)))

        # Join desired features into the features list
        if stats:
            print('\tStats features added')
            self.features += self.statistical_features
        elif ratios:
            print('\tRatio features added')
            self.features +=  self.ratios
        elif casted:
            print('\tCasted features added')
            self.features += self.casted_features

    def get_features(self, verbose=False):
        """Get features given finalized feature list"""
        if verbose:
            print('Total features: {}'.format(len(self.features)))
        return self.features

def experiment_features(data, stats=False, ratios=False, casted=False, verbose=False):
    """ Helper function for initializing ExpFeatures()

    Args:
        data (pd.DataFrame): Dataset
        stats (bool): statsTime Flag. Default is False
        ratios (bool): ratiosTime Flag. Default is False
        casted (bool): castedSessions Flag. Default is False
        verbose (bool): Verbosity Flag

    Returns:

    """
    df = data.copy()
    features_list = df.columns
    # Initialize ExpFeatures
    features = ExpFeatures(features_list, stats, ratios, casted).get_features()
    # Filter dataset for desired features
    df = df[features]
    if verbose:
        print("\nit has {0} rows and {1} columns".format(*df.shape))
        print('\n{0:*^80}\n'.format(' It has the following columns '))
        print(df.info())
    return df
