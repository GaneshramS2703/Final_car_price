import pickle
import pandas as pd

# Load the trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Get feature names used in training
feature_columns = model.feature_names_in_

# Define test input
test_data = {
    "milage": 50000,
    "model_year": 2010,
    "brand": "Audi"
}

# Convert test data to DataFrame
test_df = pd.DataFrame([test_data])

# One-Hot Encode 'brand' (match training feature names)
brand_columns = [col for col in feature_columns if col.startswith("brand_")]
test_df = pd.concat([test_df, pd.DataFrame(0, index=test_df.index, columns=brand_columns)], axis=1)

# Set the correct brand column to 1 (if it exists in training features)
brand_column = f"brand_{test_data['brand']}"
if brand_column in feature_columns:
    test_df[brand_column] = 1

# Ensure test data has all required features (fill missing ones with 0)
missing_cols = list(set(feature_columns) - set(test_df.columns))
test_df = pd.concat([test_df, pd.DataFrame(0, index=test_df.index, columns=missing_cols)], axis=1)

# Reorder columns to match training order
test_df = test_df[feature_columns]

# Make prediction
predicted_price = model.predict(test_df)[0]

# Display result
print(f"✅ Predicted Price: ${round(predicted_price, 2)}")
