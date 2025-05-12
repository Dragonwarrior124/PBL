import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# Loading and preprocessing the dataset
def preprocess_data(df):
    # Handling missing values
    df = df.copy()
    df['rear_camera_mp'] = df['rear_camera_mp'].replace('', np.nan).astype(float)
    df['rear_camera_mp'] = df['rear_camera_mp'].fillna(df['rear_camera_mp'].median())
    
    # Encoding categorical variables
    categorical_cols = ['device_brand', 'os', '4g', '5g']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Selecting features and target
    features = ['device_brand', 'os', 'screen_size', '4g', '5g', 'rear_camera_mp', 
                'front_camera_mp', 'internal_memory', 'ram', 'battery', 'weight', 
                'release_year', 'days_used']
    X = df[features]
    y = df['normalized_used_price']
    
    # Scaling numerical features
    numerical_cols = ['screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory', 
                      'ram', 'battery', 'weight', 'release_year', 'days_used']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler, label_encoders

# Main function
def main():
    # Loading the dataset
    df = pd.read_csv('used_device_data.csv')
    
    # Preprocessing
    X, y, scaler, label_encoders = preprocess_data(df)
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training the model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluating the model
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    print(f"Training R^2 Score: {train_score:.4f}")
    print(f"Testing R^2 Score: {test_score:.4f}")
    
    # Saving the model and preprocessing objects
    joblib.dump(rf_model, 'phone_price_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Model and preprocessing objects saved successfully.")

if __name__ == "__main__":
    main()