from sklearn.preprocessing import LabelEncoder
from src.data.d_utils import sample_data, preprocess_data
from src.eval_model import predict, load_model, evaluate_model

LABELS = './src/data/labels.txt'
MODEL = './models/finalized_LRmodel.sav'

def run_recommmendation(dataset, id, model_weights):
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
    return predictions, ndcg
