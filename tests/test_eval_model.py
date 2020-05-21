import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mock
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression

import src.data.d_utils
from src.eval_model import *

filename = './models/finalized_LRmodel.sav'
tst_file = './tests/LRmodel-v001_test_case.json'

class TestDefaultRecommender:

    @pytest.fixture(scope='session')
    def test_data(self):
        test_examples = pd.read_json(tst_file)
        return test_examples

    @pytest.fixture(scope='session')
    def model(self):
        return load_model(filename)

    @pytest.fixture(scope='session')
    def predictions(self):
        return [[10, 7, 11, 3, 6], [7, 10, 4, 11, 5], [7, 10, 11, 4, 6]]

    @pytest.fixture(scope='session')
    def gtruth(self):
        return [3, 7, 7]

    def test_load_model(self):
        model = load_model(filename)
        assert isinstance(model, LogisticRegression)

    def test_model_inference(self, test_data, model, predictions):
        assert predictions == predict(model, test_data)

    def test_score_predictions(self, gtruth, predictions):
        expected_score = np.array([0.43067656, 1.        , 1.        ])
        print(expected_score)
        print(score_predictions(pd.DataFrame(predictions), pd.Series(gtruth)))
        assert (expected_score == score_predictions(pd.DataFrame(predictions),
                                                    pd.Series(gtruth))).any()

    def test_evaluate_model(self, gtruth, predictions):
        metrics = {'ndcg': 0.8102255193577976}
        tested_metrics, _ = evaluate_model(gtruth, predictions)
        assert metrics == tested_metrics

    def test_plot_feature_importances(self):
        importances = np.array([0.4, 0.3, 0.8, 0.3, 0.2])
        feature_decoder = {idx: f'feature{idx}' for idx in range(len(importances))}
        plot_feature_importances(importances, feature_decoder, top_k=3, show=False)
        expected_file = 'figs_feature_importance.png'
        assert os.path.exists(expected_file)
        if os.path.exists(expected_file):
            os.remove(expected_file)

    def test_main(self):
        tst_file = './tests/test-test_users-tst.csv'
        dataset_type = 'dummy'
        csv_fnames = {f'test-{dataset_type}': tst_file}
        main(csv_fnames, filename, dataset_type, validate_model=False)
        expected_submission_csv = 'sub.csv'
        assert os.path.exists(expected_submission_csv)
        if os.path.exists(expected_submission_csv):
            os.remove(expected_submission_csv)

    def test_init(self):
        from src import eval_model
        with mock.patch.object(eval_model, "main", return_value=42):
            with mock.patch.object(eval_model, "__name__", "__main__"):
                with mock.patch.object(eval_model.sys, 'exit') as mock_exit:
                    eval_model.init()
                    assert mock_exit.call_args[0][0] == 42
