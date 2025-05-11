import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\admin\Desktop\BooksDataset.csv", encoding="latin1")
print(f"Dataset loaded. Shape: {df.shape}")

# Use a subset of data to avoid memory issues
df = df.sample(n=10000, random_state=42)
print(f"Using subset of data. New shape: {df.shape}")

# Fix column names
df.columns = [col.strip() for col in df.columns]
print("Columns fixed.")

# Extract price from "Price Starting at $x.xx"
print("Extracting prices...")
df["Price"] = df["Price"].str.extract(r"\$(\d+\.\d+|\d+)").astype(float)
print("Prices extracted.")

# Fix or rename truncated columns
if "Descriptio" in df.columns:
    df.rename(columns={"Descriptio": "Description"}, inplace=True)
if "Publish_D" in df.columns:
    df.rename(columns={"Publish_D": "Publish_Date"}, inplace=True)
print("Columns renamed.")

# Clean Publish_Date
print("Cleaning Publish_Date...")
df["Publish_Date"] = df["Publish_Date"].str.split(",", expand=True)[0].str.strip()
print("Publish_Date cleaned.")

# Drop rows with missing critical data
print("Dropping missing data...")
df = df.dropna(subset=["Title", "Authors", "Category", "Price", "Publish_Date", "Publisher"])
df = df[df["Price"] > 0]
print(f"Data cleaned. New shape: {df.shape}")

# Log-transform the target
df["Price"] = np.log1p(df["Price"])
print("Price log-transformed.")

# Create a numerical feature from Title: length of the title
print("Creating Title_Length feature...")
df["Title_Length"] = df["Title"].str.len()  # Number of characters in the title
print("Title_Length feature created.")

# Reduce cardinality of categorical columns
print("Reducing cardinality...")
for col in ["Authors", "Category", "Publish_Date", "Publisher"]:
    unique_vals = df[col].nunique()
    print(f"{col}: {unique_vals} unique values")
    if unique_vals > 100:
        top_categories = df[col].value_counts().index[:50]  # Keep top 50
        df[col] = df[col].apply(lambda x: x if x in top_categories else "Other")
        print(f"Reduced {col} to {df[col].nunique()} unique values")

# Encode categorical features and include Title_Length
print("Encoding categorical features...")
X = pd.get_dummies(df[["Authors", "Category", "Publish_Date", "Publisher"]], drop_first=True)
X["Title_Length"] = df["Title_Length"]  # Add the numerical feature
y = df["Price"]
print(f"Encoded features shape: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train model using Linear Regression
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained.")

# Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained. Test MSE: {mse:.4f}, R²: {r2:.4f}")

# Save model and feature columns
joblib.dump(model, "model_linear.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("✅ Model saved as model_linear.pkl")