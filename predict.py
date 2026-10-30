"""
Prediction script for classification model

This script loads a trained model and makes predictions on new data.
"""

import argparse
import pandas as pd
import pickle

parser = argparse.ArgumentParser(description="Run predictions with optional SHAP explanations")
parser.add_argument("--input", default="new_data.csv", help="Path to input CSV")
parser.add_argument("--explain-row", type=int, default=-1, help="Row index to explain with SHAP (-1 disables)")
args = parser.parse_args()

# Load the trained model
with open('classification_model_20261030_164534.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    scaler = saved_data.get('scaler')
    label_encoder = saved_data.get('label_encoder')
    feature_names = saved_data.get('feature_names')
    text_column = saved_data.get('text_column')
    is_text_model = saved_data.get('is_text_model', False)

# Load new data for prediction
# TODO: Update this path to your new data
new_data = pd.read_csv(args.input)

# Prepare data (same preprocessing as training)
if is_text_model and text_column and text_column in new_data.columns:
    X_input = new_data[text_column].fillna("").astype(str)
    predictions = model.predict(X_input)
else:
    X = pd.get_dummies(new_data, drop_first=True)
    X = X.fillna(X.mean())

    # Align columns with training features when available
    if feature_names:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

    # Scale features when scaler is available
    X_input = scaler.transform(X) if scaler is not None else X
    predictions = model.predict(X_input)

# Decode if classification with label encoder
if label_encoder is not None:
    predictions = label_encoder.inverse_transform(predictions)

# Save predictions
result_df = pd.DataFrame({
    'predictions': predictions
})
result_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
print("\nFirst few predictions:")
print(result_df.head())

# Optional SHAP explanation for one prediction
if args.explain_row >= 0:
    try:
        import shap
        if is_text_model:
            raise ValueError("Row-level SHAP in predict.py is currently supported for tabular models only.")
        row_idx = min(args.explain_row, len(X) - 1)
        single_row = X.iloc[[row_idx]]
        explainer = shap.Explainer(model, X)
        explanation = explainer(single_row)
        shap_values = explanation.values.tolist()
        print(f"\nSHAP explanation generated for row {row_idx}")
        print("SHAP values:", shap_values)
    except Exception as e:
        print(f"\nSHAP explanation skipped: {e}")
