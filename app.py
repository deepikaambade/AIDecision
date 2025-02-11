from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load MLP model
MODEL_PATH = "mlp_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)  # Handle missing values
    df.drop(columns=["UDI", "Product ID"], errors="ignore", inplace=True)  # Remove unnecessary columns
    return df

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess & predict
    input_data = preprocess_data(file_path)
    predictions = model.predict(input_data)
    
    # Return results
    results = {"predictions": predictions.tolist()}
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
