import pandas as pd
import joblib
import os

# Project directory and file paths
PROJECT_DIR = r"D:\project"
MODEL_PATH = os.path.join(PROJECT_DIR, "price_model.pkl")
LE_BRAND_PATH = os.path.join(PROJECT_DIR, "le_brand.pkl")
LE_PROCESSOR_PATH = os.path.join(PROJECT_DIR, "le_processor.pkl")
LE_RAM_TYPE_PATH = os.path.join(PROJECT_DIR, "le_ram_type.pkl")
LE_ROM_TYPE_PATH = os.path.join(PROJECT_DIR, "le_rom_type.pkl")
LE_GPU_PATH = os.path.join(PROJECT_DIR, "le_gpu.pkl")
LE_OS_PATH = os.path.join(PROJECT_DIR, "le_os.pkl")
SCALER_PATH = os.path.join(PROJECT_DIR, "scaler.pkl")

# Function to check if model, encoders, and scaler exist
def model_exists():
    return (os.path.exists(MODEL_PATH) and 
            os.path.exists(LE_BRAND_PATH) and 
            os.path.exists(LE_PROCESSOR_PATH) and 
            os.path.exists(LE_RAM_TYPE_PATH) and 
            os.path.exists(LE_ROM_TYPE_PATH) and 
            os.path.exists(LE_GPU_PATH) and 
            os.path.exists(LE_OS_PATH) and 
            os.path.exists(SCALER_PATH))

# Function to predict price for a new item
def predict_price(brand, spec_rating, processor, ram, ram_type, rom, rom_type, gpu, display_size, resolution_width, resolution_height, os, warranty):
    if not model_exists():
        print("Error: Model not found. Please run train_model.py first.")
        return None

    # Load model, encoders, and scaler
    model = joblib.load(MODEL_PATH)
    le_brand = joblib.load(LE_BRAND_PATH)
    le_processor = joblib.load(LE_PROCESSOR_PATH)
    le_ram_type = joblib.load(LE_RAM_TYPE_PATH)
    le_rom_type = joblib.load(LE_ROM_TYPE_PATH)
    le_gpu = joblib.load(LE_GPU_PATH)
    le_os = joblib.load(LE_OS_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Validate categorical inputs
    categorical_inputs = {
        'brand': (brand, le_brand),
        'processor': (processor, le_processor),
        'Ram_type': (ram_type, le_ram_type),
        'ROM_type': (rom_type, le_rom_type),
        'GPU': (gpu, le_gpu),
        'OS': (os, le_os)
    }
    
    encoded_inputs = {}
    for feature, (value, encoder) in categorical_inputs.items():
        try:
            encoded_inputs[feature] = encoder.transform([value])[0]
        except ValueError:
            print(f"Error: {feature} '{value}' not recognized. Available options: {encoder.classes_}")
            return None

    # Prepare numerical inputs
    numerical_inputs = {
        'spec_rating': spec_rating,
        'Ram': ram,
        'ROM': rom,
        'display_size': display_size,
        'resolution_width': resolution_width,
        'resolution_height': resolution_height,
        'warranty': warranty
    }

    # Create input dataframe
    input_data = pd.DataFrame({
        'spec_rating': [numerical_inputs['spec_rating']],
        'Ram': [numerical_inputs['Ram']],
        'ROM': [numerical_inputs['ROM']],
        'display_size': [numerical_inputs['display_size']],
        'resolution_width': [numerical_inputs['resolution_width']],
        'resolution_height': [numerical_inputs['resolution_height']],
        'warranty': [numerical_inputs['warranty']],
        'brand': [encoded_inputs['brand']],
        'processor': [encoded_inputs['processor']],
        'Ram_type': [encoded_inputs['Ram_type']],
        'ROM_type': [encoded_inputs['ROM_type']],
        'GPU': [encoded_inputs['GPU']],
        'OS': [encoded_inputs['OS']]
    })

    # Scale numerical features
    numerical_features = ['spec_rating', 'Ram', 'ROM', 'display_size', 'resolution_width', 'resolution_height', 'warranty']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Predict price
    price = model.predict(input_data)[0]
    return price

# Main function for user interaction
def main():
    print("=== Laptop Price Prediction ===")
    while True:
        print("\nOptions:")
        print("1. Predict price")
        print("2. Exit")
        choice = input("Enter choice (1-2): ")

        if choice == '1':
            print("\nEnter laptop details:")
            brand = input("Brand (e.g., HP, Apple): ")
            try:
                spec_rating = float(input("Spec Rating (e.g., 73.0): "))
                ram = float(input("RAM (GB, e.g., 8): "))
                rom = float(input("Storage (GB, e.g., 512 or 1000 for 1TB): "))
                display_size = float(input("Display Size (inches, e.g., 15.6): "))
                resolution_width = float(input("Resolution Width (pixels, e.g., 1920): "))
                resolution_height = float(input("Resolution Height (pixels, e.g., 1080): "))
                warranty = int(input("Warranty (years, e.g., 1): "))
            except ValueError:
                print("Error: Numerical inputs must be numbers.")
                continue
            processor = input("Processor (e.g., 5th Gen AMD Ryzen 5 5600H): ")
            ram_type = input("RAM Type (e.g., DDR4): ")
            rom_type = input("Storage Type (e.g., SSD): ")
            gpu = input("GPU (e.g., Intel Iris Xe Graphics): ")
            os = input("OS (e.g., Windows 11 OS): ")

            price = predict_price(
                brand, spec_rating, processor, ram, ram_type, rom, rom_type, 
                gpu, display_size, resolution_width, resolution_height, os, warranty
            )
            if price is not None:
                print(f"\nRecommended selling price: ₹{price:.2f}")

        elif choice == '2':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()