import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.train_model import *

class TestTrainModel():

    def test_main(self):
        dataset_type = 'dummy'
        CSV_FNAMES = {f'train-{dataset_type}': './tests/test-train_users_2-tst.csv',
                      f'test-{dataset_type}': './tests/test-test_users-tst.csv'}
        main(CSV_FNAMES, dataset_type, xgb_model=False, save=True)
        expected_output_file = 'sub.csv'
        assert os.path.exists(expected_output_file)
        if os.path.exists(expected_output_file):
            os.remove(expected_output_file)

        expected_model_weights = 'finalized_LRmodel.sav'
        assert os.path.exists(expected_model_weights)
        if os.path.exists(expected_model_weights):
            os.remove(expected_model_weights)
