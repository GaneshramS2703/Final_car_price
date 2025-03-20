from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Get feature names used in training
feature_columns = model.feature_names_in_

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure JSON data is received correctly
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Extract input values
        milage = data.get("milage")
        model_year = data.get("model_year")
        brand = data.get("brand")

        # Validate input data
        if milage is None or model_year is None or brand is None:
            return jsonify({"error": "Missing required parameters (milage, model_year, brand)"}), 400

        # Create DataFrame with required features
        test_df = pd.DataFrame([{"milage": milage, "model_year": model_year}])

        # One-Hot Encode 'brand' (initialize all brand columns to 0)
        brand_columns = [col for col in feature_columns if col.startswith("brand_")]
        brand_df = pd.DataFrame(0, index=[0], columns=brand_columns)
        
        # Set the correct brand column to 1 (if it exists in training features)
        brand_column = f"brand_{brand}"
        if brand_column in feature_columns:
            brand_df[brand_column] = 1

        # Combine all required features
        test_df = pd.concat([test_df, brand_df], axis=1)

        # Ensure test data has all required features
        missing_cols = list(set(feature_columns) - set(test_df.columns))
        for col in missing_cols:
            test_df[col] = 0  # Fill missing columns with 0

        # Reorder columns to match model training order
        test_df = test_df[feature_columns]

        # Make prediction
        predicted_price = model.predict(test_df)[0]

        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
