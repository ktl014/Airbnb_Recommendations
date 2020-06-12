""" Recommendation module

Module for running model recommendation
To execute a recommendation, module expects an input of the desired dataset and model
weights. Loading the dataset and model will then take place. Inference is done via
modules from `eval_model`. Postprocessing is the final step.

We specifically use this within our streamlit applications to make recommendation calls.

"""
from sklearn.preprocessing import LabelEncoder
from src.data.d_utils import sample_data, preprocess_data
from src.eval_model import predict, load_model, evaluate_model

LABELS = './src/data/labels.txt'
MODEL = './models/finalized_LRmodel.sav'

COUNTRY_ABB2NAME = {
 'US': 'USA',
 'FR': 'France',
 'CA': 'Canada',
 'GB': 'United Kingdom',
 'ES': 'Spain',
 'IT': 'Italy',
 'PT': 'Portugal',
 'NL': 'Netherlands',
 'DE': 'Germany',
 'AU': 'Australia',
 'NDF': 'No Destination Found',
 'other': 'Other'}

def run_recommmendation(dataset, id, model_weights, full_name=False):
    """ Run model recommendations

    Usage

    >>> from recommend import run_recommmendation
    >>> dataset = load_data('train_users_2.csv', features=True)
    >>> MODEL = './models/finalized_LRmodel.sav'
    >>> predictions, ndcg = run_recommmendation(dataset, '87mebub9p4', model_weights=MODEL)
    {'ndcg': 0.6309297535714575}

    Args:
        dataset (Airbnb): Named tuple collection, Airbnb
        id (str): User ID
        model_weights (str): Model path weights
        full_name (bool): Flag for country full name

    Returns:

    """
    le = LabelEncoder().fit(open(LABELS).read().splitlines())

    d, id = sample_data(dataset.users_feat, id=id)
    X, y = preprocess_data(d)

    # Load model
    model = load_model(model_weights)
    # Make model inference
    # then postprocess the prediction
    predictions = predict(model, X)
    ndcg, score = evaluate_model(y, predictions)
    predictions = le.inverse_transform(predictions[0])
    if full_name:
        predictions = [COUNTRY_ABB2NAME[pred] for pred in predictions]
    return predictions, ndcg
