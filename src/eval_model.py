"""Evaluate the model"""

# Standard dist
import os
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Third party imports
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Project level imports
from src.data.d_utils import read_in_dataset

# Module level constants
DATA_DIR = '../airbnb-recruiting-new-user-bookings'

CSV_FNAMES = {
    'train': os.path.join(DATA_DIR, 'train_users_2.csv'),
    'train-processed': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'test-processed': os.path.join(DATA_DIR, 'test_users-processed.csv'),
    'train-feat_eng': os.path.join(DATA_DIR, 'train_users_2-feature_eng.csv'),
    'test-feat_eng': os.path.join(DATA_DIR, 'test_users-feature_eng.csv'),
    'train-merged_sessions': os.path.join(DATA_DIR, 'train_users-merged_sessions.csv'),
    'test-merged_sessions': os.path.join(DATA_DIR, 'test_users-merged_sessions.csv'),
    'train-part-merged_sessions': os.path.join(DATA_DIR, 'train_users-part-merged_sessions.csv'),
    'val': os.path.join(DATA_DIR, 'val_users.csv'),
    'val-part-merged_sessions': os.path.join(DATA_DIR, 'val_users-part-merged_sessions.csv')
}

VALIDATE_MODEL = True

def main():
    # load the model from disk
    filename = '../finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Set up dataset
    # DATASET_TYPE = 'processed'
    # DATASET_TYPE = 'feat_eng'
    DATASET_TYPE = 'merged_sessions'
    class AirBnB(): pass
    airbnb = AirBnB()

    airbnb.X_test = read_in_dataset(CSV_FNAMES['test-{}'.format(DATASET_TYPE)],
                                    keep_id=True, verbose=True)
    airbnb.test_id = airbnb.X_test.pop('id')
    airbnb.X_test.pop('country_destination')

    if VALIDATE_MODEL:
        # Load dataset
        DATASET_TYPE = 'part-merged_sessions'
        airbnb.X = read_in_dataset(CSV_FNAMES['val-{}'.format(DATASET_TYPE)],
                                   keep_id=False, verbose=True)
        airbnb.y = airbnb.X.pop('country_destination')

        prob = loaded_model.predict_proba(airbnb.X)
        airbnb.y_pred = top_k_predictions(prob, k=5)
        score = score_predictions(pd.DataFrame(airbnb.y_pred), pd.Series(airbnb.y))
        final_score = np.mean(score)
        print(final_score)

    # Run model on test set
    airbnb.y_pred_test = loaded_model.predict_proba(airbnb.X_test)

    # Write test predictions to submission file
    # Taking the 5 classes with highest probabilities
    ids = []  # list of ids
    cts = []  # list of countries
    for i in range(len(airbnb.test_id)):
        idx = airbnb.test_id[i]
        ids += [idx] * 5
        cts += airbnb.le.inverse_transform(np.argsort(airbnb.y_pred_test[i])[::-1])[:5].tolist()
    print('complete')
    # Generate submission
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    sub.to_csv('sub.csv', index=False)

def load_model(fname):
    return pickle.load(open(fname, 'rb'))

"""Evaluation Metric Functions"""
def plot_feature_importances(importances, feature_decoder, top_k=10):
    """Plot feature importances for XGB/Random Forest model"""
    import matplotlib.pyplot as plt
    import numpy as np
    sorted_idx = np.argsort(importances)[::-1]
    if top_k:
        sorted_idx = sorted_idx[:top_k]
    importances = importances[sorted_idx]
    decoded_feature_names = [feature_decoder[idx] for idx in sorted_idx]
    plt.barh(range(len(importances)), importances[::-1])
    plt.yticks(range(len(importances)), decoded_feature_names[::-1])
    plt.xlabel('Importance values')
    plt.ylabel('Features')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('figs_feature_importance.png', bbox_inches = "tight")
    plt.show()

def evaluate_model(gtruth, predictions, verbose=True, normalize=True, beta=0):
    """Compute all relevant evaluation metrics for a given model"""
    metrics = {}
    score = score_predictions(pd.DataFrame(predictions), pd.Series(gtruth))
    metrics['ndcg'] = np.mean(score)

    return metrics, score

def top_k_predictions(pred,k):
    return [np.argsort(pred[i])[::-1][:k].tolist() for i in range(len(pred))]

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def score_predictions(preds, truth, n_modes=5):
    """
    preds: pd.DataFrame
      one row for each observation, one column for each prediction.
      Columns are sorted from left to right descending in order of likelihood.
    truth: pd.Series
      one row for each obeservation.
    """
    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0

    score = np.array(r.apply(ndcg_at_k, axis=1, result_type='reduce'))
    return score

def predict(model, X):
    prob = model.predict_proba(X)
    return top_k_predictions(prob, k=5)


def init():
  if __name__ == "__main__":
    sys.exit(main())

init()