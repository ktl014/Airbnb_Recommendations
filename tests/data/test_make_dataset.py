import os
import sys
from pathlib import Path
print(str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.data.make_dataset import *

class TestMakeDataset():
    @pytest.fixture(scope='session')
    def csv_fnames(self):
        return {'train': './tests/data/train_usrs-make_dataset.csv',
                'test': './tests/data/test_usrs-make_dataset.csv'}

    def test_Airbnb_init(self, csv_fnames):
        data = pd.read_csv(csv_fnames['train'])
        dataset = AirBnBDataset(data, process_data=True)
        assert hasattr(dataset, 'data')
        assert dataset.data.shape == (25, 59)

    def test_make_dataset_main_baseline(self, csv_fnames):
        # assert datasets are saved
        main(csv_fnames, do_baseline=True, do_merged_sessions=False)
        expected_output_csv = csv_fnames['train'].split('.csv')[0] + '-processed.csv'
        assert os.path.exists(expected_output_csv)
        if os.path.exists(expected_output_csv):
            os.remove(expected_output_csv)

        expected_output_csv = csv_fnames['test'].split('.csv')[0] + '-processed.csv'
        assert os.path.exists(expected_output_csv)
        if os.path.exists(expected_output_csv):
            os.remove(expected_output_csv)

    def test_make_dataset_main_airbnb(self, csv_fnames):
        main(csv_fnames, do_baseline=False, do_merged_sessions=False)
        expected_output_csv = csv_fnames['train'].split('.csv')[0] + '-feature_eng.csv'
        assert os.path.exists(expected_output_csv)
        if os.path.exists(expected_output_csv):
            os.remove(expected_output_csv)

        expected_output_csv = csv_fnames['test'].split('.csv')[0] + '-feature_eng.csv'
        assert os.path.exists(expected_output_csv)
        if os.path.exists(expected_output_csv):
            os.remove(expected_output_csv)
