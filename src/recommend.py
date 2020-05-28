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
