import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load saved feature lists
with open('./model/features/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('./model/features/cat_features.pkl', 'rb') as f:
    cat_features = pickle.load(f)

# Load trained models dict with keys 'catboost_models' and 'lgb_models'
with open('./model/house_price_predict_model.pkl', 'rb') as f:
    house_price_predict_model = pickle.load(f)

cat_models = house_price_predict_model['catboost_models']
lgb_models = house_price_predict_model['lgb_models']

def preprocess_input(input_json):
    df = pd.DataFrame([input_json])

    # Fill missing features
    for feat in features:
        if feat not in df.columns:
            if feat in cat_features:
                df[feat] = ''
            else:
                df[feat] = 0

    df = df[features]

    # Convert categorical columns to category dtype
    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype('category')  # This is key!

    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X = preprocess_input(data)
        
        # Debug print statements to check dtypes and sample values
        print(X.dtypes)
        print(X.head())

        preds_cb = np.zeros(len(X))
        for model in cat_models:
            preds_cb += np.expm1(model.predict(X)) / len(cat_models)

        preds_lgb = np.zeros(len(X))
        for model in lgb_models:
            preds_lgb += np.expm1(model.predict(X, num_iteration=model.best_iteration)) / len(lgb_models)

        # Weighted blend of predictions
        final_preds = 0.6 * preds_cb + 0.4 * preds_lgb

        result = final_preds[0] if len(final_preds) == 1 else final_preds.tolist()

        return jsonify({'predicted_price': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
