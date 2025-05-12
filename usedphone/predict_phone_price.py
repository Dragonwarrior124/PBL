import pandas as pd
import numpy as np
import joblib

# Loading the model and preprocessing objects
rf_model = joblib.load('phone_price_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Function to preprocess user input
def preprocess_user_input(user_input):
    input_df = pd.DataFrame([user_input])
    
    # Encoding categorical variables
    categorical_cols = ['device_brand', 'os', '4g', '5g']
    for col in categorical_cols:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform([input_df[col].iloc[0]])[0]
        except ValueError:
            print(f"Error: {col} value not seen during training. Using default encoding.")
            input_df[col] = le.transform([le.classes_[0]])[0]
    
    # Selecting features
    features = ['device_brand', 'os', 'screen_size', '4g', '5g', 'rear_camera_mp', 
                'front_camera_mp', 'internal_memory', 'ram', 'battery', 'weight', 
                'release_year', 'days_used']
    X = input_df[features]
    
    # Scaling numerical features
    numerical_cols = ['screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory', 
                      'ram', 'battery', 'weight', 'release_year', 'days_used']
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X

# Main function
def main():
    print("Enter the phone details to predict its used price:")
    
    # Collecting user input
    user_input = {}
    user_input['device_brand'] = input("Device Brand (e.g., Honor, Samsung, Apple): ")
    user_input['os'] = input("OS (e.g., Android, iOS, Others): ")
    user_input['screen_size'] = float(input("Screen Size (cm, e.g., 15.37): "))
    user_input['4g'] = input("4G (yes/no): ").lower()
    user_input['5g'] = input("5G (yes/no): ").lower()
    user_input['rear_camera_mp'] = float(input("Rear Camera MP (e.g., 13): "))
    user_input['front_camera_mp'] = float(input("Front Camera MP (e.g., 8): "))
    user_input['internal_memory'] = float(input("Internal Memory (GB, e.g., 64): "))
    user_input['ram'] = float(input("RAM (GB, e.g., 4): "))
    user_input['battery'] = float(input("Battery (mAh, e.g., 4000): "))
    user_input['weight'] = float(input("Weight (g, e.g., 180): "))
    user_input['release_year'] = int(input("Release Year (e.g., 2020): "))
    user_input['days_used'] = int(input("Days Used (e.g., 200): "))
    
    # Preprocessing input
    X = preprocess_user_input(user_input)
    
    # Making prediction and converting to actual price
    predicted_normalized_price = rf_model.predict(X)[0]
    actual_price = np.exp(predicted_normalized_price)  # Reverse log-transformation
    print(f"\nPredicted Used Phone Price: ~${actual_price:.2f}")

if __name__ == "__main__":
    main()