import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature columns
print("Loading model and feature columns...")
model = joblib.load("model_linear.pkl")
feature_columns = joblib.load("feature_columns.pkl")
print("Model and feature columns loaded successfully.")

# Take user input for book details
print("\nEnter the book details:")
title = input("Book Title: ")
authors = input("Authors: ")
category = input("Category (e.g., Fiction, Self-Help): ")
publish_date = input("Publish Date (e.g., Monday): ")
publisher = input("Publisher: ")

# Create a DataFrame with the user input
new_data = pd.DataFrame({
    "Title": [title],
    "Authors": [authors],
    "Category": [category],
    "Publish_Date": [publish_date],
    "Publisher": [publisher]
})
print("\nUser input data:")
print(new_data)

# Create Title_Length feature
new_data["Title_Length"] = new_data["Title"].str.len()

# Encode the categorical features
print("Encoding input data...")
new_data_encoded = pd.get_dummies(new_data[["Authors", "Category", "Publish_Date", "Publisher"]], drop_first=True)

# Add Title_Length to the encoded data
new_data_encoded["Title_Length"] = new_data["Title_Length"]

# Align the encoded data with the training feature columns
new_data_aligned = pd.DataFrame(0, index=new_data_encoded.index, columns=feature_columns)
for col in new_data_encoded.columns:
    if col in feature_columns:
        new_data_aligned[col] = new_data_encoded[col]
print(f"Encoded and aligned data shape: {new_data_aligned.shape}")

# Make predictions (model outputs log-transformed prices)
print("\nMaking prediction...")
predicted_log_price = model.predict(new_data_aligned)
print("Prediction (log-transformed):", predicted_log_price[0])

# Reverse the log-transformation to get actual price
predicted_price = np.expm1(predicted_log_price[0])  # Reverse of log1p
print(f"Predicted price for '{title}': ${predicted_price:.2f}")