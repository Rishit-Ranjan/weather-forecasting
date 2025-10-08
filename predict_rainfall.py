import pandas as pd
import joblib
import os

# Load the scaler, trained model, and model columns
if not all(os.path.exists(f) for f in ['scaler.pkl', 'rainfall_predictor_model.pkl', 'model_columns.pkl']):
    print("Error: Model files not found. Please run train_model.py first.")
    exit()

scaler = joblib.load('scaler.pkl')
model = joblib.load('rainfall_predictor_model.pkl')
model_columns = joblib.load('model_columns.pkl')

def predict_rainfall(new_data):
    """Predicts rainfall for new weather data."""
    # Convert input into DataFrame
    new_df = pd.DataFrame([new_data])

    # Align columns with the training data, filling missing with 0
    # This is crucial for handling one-hot encoded features that might not be in the new data
    new_df = new_df.reindex(columns=model_columns, fill_value=0)

    # List of numeric features to scale
    features_to_scale = ['temperature', 'humidity', 'wind_speed', 'pressure', 'hour', 'dayofweek']
    
    # Scale numeric features using the saved scaler
    new_df[features_to_scale] = scaler.transform(new_df[features_to_scale])

    # Predict rainfall using the trained model
    predicted_rainfall = model.predict(new_df)

    return predicted_rainfall[0]

def main():
    """Main function to demonstrate a prediction."""
    # Example new input data (replace with actual input)
    # Note: For a real application, you would get this data from an API or user input.
    # The weather descriptions must match those encountered during training.
    new_data = {
        'temperature': 22.5,
        'humidity': 60,
        'wind_speed': 3.0,
        'pressure': 1012,
        'hour': 14,
        'dayofweek': 2, # Wednesday
        # Example of one-hot encoded feature. 
        # If the weather is 'scattered clouds', this column is 1, others are 0.
        'weather_scattered clouds': 1 
    }

    prediction = predict_rainfall(new_data)
    print(f"Predicted Rainfall (3-hour interval, mm): {prediction:.4f}")

if __name__ == "__main__":
    main()
