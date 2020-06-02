""" Model Selection script via CrossValidation

Model selection is done over a mixture of discriminative and generative models
"""
# Standard dist imports
import os
import datetime
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

# Third party imports
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data.d_utils import read_in_dataset

# Module level constants
# Select dataset type
DATASET_TYPE = 'processed'
# DATASET_TYPE = 'feat_eng'
# DATASET_TYPE = 'merged_sessions'
DATA_DIR = '../airbnb-recruiting-new-user-bookings'
CSV_FNAMES = {
    'train-processed': os.path.join(DATA_DIR, 'train_users_2-processed.csv'),
    'test-processed': os.path.join(DATA_DIR, 'test_users-processed.csv'),
    'train-feat_eng': os.path.join(DATA_DIR, 'train_users_2-feature_eng.csv'),
    'test-feat_eng': os.path.join(DATA_DIR, 'test_users-feature_eng.csv'),
    'train-merged_sessions': os.path.join(DATA_DIR, 'train_users-merged_sessions.csv'),
    'test-merged_sessions': os.path.join(DATA_DIR, 'test_users-merged_sessions.csv')
}

MODELS = [
    DummyClassifier(strategy='most_frequent'),
    LogisticRegression(solver='lbfgs', multi_class='auto'),
    RandomForestClassifier(),
    XGBClassifier(),
    SVC(),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=2)
]

def main(csv_fnames=CSV_FNAMES, dataset_type=DATASET_TYPE, models=MODELS):
    """ Main function for executing model selection

    Args:
        csv_fnames (dict): Dictionary of csv absolute paths
        dataset_type (str): Dataset type
        models (list): List of scikit learn models

    Returns:

    """
    # Read in training set and encode labels
    class AirBnB():
        pass

    airbnb = AirBnB()
    airbnb.X = read_in_dataset(csv_fnames['train-{}'.format(dataset_type)], verbose=True)
    airbnb.idx2feature = {idx: feature for idx, feature in enumerate(airbnb.X.columns)}
    airbnb.feature2idx = {feature: idx for idx, feature in enumerate(airbnb.X.columns)}
    airbnb.train_labels = airbnb.X.pop('country_destination')
    airbnb.le = LabelEncoder()
    airbnb.le.fit(airbnb.train_labels)
    airbnb.target_labels = airbnb.le.classes_
    airbnb.y = airbnb.le.transform(airbnb.train_labels)

    # Partition and split datasets
    SEED = 42
    PARTITION = 0.10
    airbnb.X_train, airbnb.X_val, \
    airbnb.y_train, airbnb.y_val = train_test_split(airbnb.X, airbnb.y,
                                                    test_size=PARTITION, shuffle=True,
                                                    random_state=SEED)

    feature_names = list(airbnb.X_train.columns)
    for model in models:
        try:
            training_cv_score_model(airbnb.X_train, airbnb.y_train, model, feature_names)
        except Exception as e:
            raise e
    return 0

def training_cv_score_model(X, y, model, feature_names, n_folds=10):
    """
    This function takes the design matrix and target vector of the training set,
    along with a classification estimator and computes a 10 fold cross validated
    mean and standard deviation based on balanced accuracy.
    This score is printed to the end user.
    """
    numeric_transformer = Pipeline(steps=[
        ('scale_x_num', StandardScaler())
    ])

    pre_processor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, feature_names)
    ])

    clf = Pipeline(steps=[
        ('preprocessor', pre_processor),
        ('classifier', model)
    ])

    scores = cross_val_score(clf,
                             X,
                             y,
                             scoring='balanced_accuracy',
                             cv=n_folds,
                             verbose=2)

    mean_score = scores.mean()
    avg_score = scores.std()

    model_name = clf.named_steps['classifier'].__class__.__name__

    print(f'{model_name}, Average Score : {mean_score} & Standard Deviation: {avg_score}')
    print('-'*90)

    return (model_name, mean_score, avg_score)

if __name__ == '__main__':
    main()
