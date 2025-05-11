import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Project directory and file paths
PROJECT_DIR = r"D:\project"
DATA_PATH = os.path.join(PROJECT_DIR, "data.csv")
MODEL_PATH = os.path.join(PROJECT_DIR, "price_model.pkl")
LE_BRAND_PATH = os.path.join(PROJECT_DIR, "le_brand.pkl")
LE_PROCESSOR_PATH = os.path.join(PROJECT_DIR, "le_processor.pkl")
LE_RAM_TYPE_PATH = os.path.join(PROJECT_DIR, "le_ram_type.pkl")
LE_ROM_TYPE_PATH = os.path.join(PROJECT_DIR, "le_rom_type.pkl")
LE_GPU_PATH = os.path.join(PROJECT_DIR, "le_gpu.pkl")
LE_OS_PATH = os.path.join(PROJECT_DIR, "le_os.pkl")

def train_model():
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please create a CSV file with columns: brand, spec_rating, processor, Ram, Ram_type, ROM, ROM_type, GPU, display_size, resolution_width, resolution_height, OS, warranty, price")
        return False

    data = pd.read_csv(DATA_PATH)
    
    # Check required columns
    required_columns = ['brand', 'spec_rating', 'processor', 'Ram', 'Ram_type', 'ROM', 'ROM_type', 'GPU', 'display_size', 'resolution_width', 'resolution_height', 'OS', 'warranty', 'price']
    if not all(col in data.columns for col in required_columns):
        print(f"Error: Dataset must contain columns: {required_columns}")
        return False

    # Preprocess data
    # Extract numerical values from Ram and ROM (e.g., '8GB' -> 8, '1TB' -> 1000)
    data['Ram'] = data['Ram'].str.extract(r'(\d+)').astype(float)
    data['ROM'] = data['ROM'].apply(lambda x: float(x.replace('TB', '000').replace('GB', '')) if isinstance(x, str) else x)

    # Define features
    numerical_features = ['spec_rating', 'Ram', 'ROM', 'display_size', 'resolution_width', 'resolution_height', 'warranty']
    categorical_features = ['brand', 'processor', 'Ram_type', 'ROM_type', 'GPU', 'OS']
    features = numerical_features + categorical_features
    target = 'price'

    # Handle missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[numerical_features] = num_imputer.fit_transform(data[numerical_features])
    data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

    # Encode categorical features
    encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        encoders[feature] = le

    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Features and target
    X = data[features]
    y = data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation: MAE = ₹{mae:.2f}, R² = {r2:.2f}")

    # Save model, encoders, and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders['brand'], LE_BRAND_PATH)
    joblib.dump(encoders['processor'], LE_PROCESSOR_PATH)
    joblib.dump(encoders['Ram_type'], LE_RAM_TYPE_PATH)
    joblib.dump(encoders['ROM_type'], LE_ROM_TYPE_PATH)
    joblib.dump(encoders['GPU'], LE_GPU_PATH)
    joblib.dump(encoders['OS'], LE_OS_PATH)
    joblib.dump(scaler, os.path.join(PROJECT_DIR, "scaler.pkl"))
    print(f"Model, encoders, and scaler saved to {PROJECT_DIR}")
    return True

if __name__ == "__main__":
    # Ensure project directory exists
    if not os.path.exists(PROJECT_DIR):
        os.makedirs(PROJECT_DIR)
    
    print("=== Training Pricing Model ===")
    if train_model():
        print("Training completed successfully!")
    else:
        print("Training failed. Check dataset and try again.")