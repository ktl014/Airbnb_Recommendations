import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.data.d_utils import read_in_dataset
from src.model.select_model import *

dataset_type = 'dummy'
CSV_FNAMES = {f'train-{dataset_type}': './tests/test-train_users_2-tst.csv',
              f'test-{dataset_type}': './tests/test-test_users-tst.csv'}
LABELS = './src/data/labels.txt'

class TestSelectModel():
    def test_main(self):
        models = [DummyClassifier(strategy='most_frequent'),
                  LogisticRegression(solver='lbfgs', multi_class='auto')]
        assert 0 == main(csv_fnames=CSV_FNAMES, dataset_type=dataset_type, models=models)

    def test_training_cv_score_model(self):
        model = DummyClassifier(strategy='most_frequent')
        le = LabelEncoder()
        le.fit(open(LABELS).read().splitlines())
        X = read_in_dataset(CSV_FNAMES[f'train-{dataset_type}'])
        train_labels = X.pop('country_destination')
        y = le.transform(train_labels)

        feature_names = list(X.columns)
        expected_score = ('DummyClassifier', 0.275, 0.03818813079129866)
        score = training_cv_score_model(X, y, model, feature_names)
        assert expected_score == score
