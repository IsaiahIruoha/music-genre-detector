import joblib
import pandas as pd
import numpy as np

rf_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/rf_best_model.pkl')
svm_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/svm_best_model.pkl')
gb_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/gb_best_model.pkl')
knn_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/knn_best_model.pkl')
logreg_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/logreg_best_model.pkl')
scaler = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/scaler.pkl')
encoder = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/encoder.pkl')

# Define a function to evaluate a model on unlabeled external data
def evaluate_model_on_external_data(model, X_ext):
    y_pred = model.predict(X_ext)
    return y_pred

# Function to predict genre
def predict_genre(features):
    # Ensure all columns except 'filename', 'start', and 'end' are converted to numeric
    numeric_features = features.drop(columns=['filename', 'start', 'end']).apply(pd.to_numeric, errors='coerce')
    
    # Replace any NaNs that might have resulted from coercion to numeric
    numeric_features = numeric_features.fillna(0)
    
    # Scale the features
    X_ext = scaler.transform(numeric_features)
    
    # Evaluate each model and gather predictions
    rf_predictions = evaluate_model_on_external_data(rf_best_model, X_ext)
    svm_predictions = evaluate_model_on_external_data(svm_best_model, X_ext)
    gb_predictions = evaluate_model_on_external_data(gb_best_model, X_ext)
    knn_predictions = evaluate_model_on_external_data(knn_best_model, X_ext)
    logreg_predictions = evaluate_model_on_external_data(logreg_best_model, X_ext)

    # Aggregate predictions
    all_predictions = np.vstack([rf_predictions, svm_predictions, gb_predictions, knn_predictions, logreg_predictions])
    final_predictions = [np.bincount(row).argmax() for row in all_predictions.T]

    # Map predicted numbers to genre names
    final_predictions_genre = [encoder.classes_[pred] for pred in final_predictions]
    
    return final_predictions_genre


# Example usage
# features = pd.read_csv('path_to_csv_file.csv')
# predicted_genres = predict_genre(features)
# print(predicted_genres)
