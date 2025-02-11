import argparse
import pandas as pd
import joblib
import os
import numpy as np
import traceback
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load Data
def load_data(file_path):
    try:
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"Original DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Drop rows with any NaN values
        df = df.dropna()
        print(f"DataFrame shape after dropping NaNs: {df.shape}")
        
        # Encode categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            print(f"Encoding column: {col}")
            df[col] = LabelEncoder().fit_transform(df[col])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print(traceback.format_exc())
        raise

# Train Model
def train_model(train_file, model_file):
    try:
        df = load_data(train_file)
        
        # Separate features and target
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target column name: {df.columns[-1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = MLPClassifier(
            hidden_layer_sizes=(50, 25), 
            max_iter=500, 
            random_state=42,
            verbose=True  # Added verbose for more training details
        )
        model.fit(X_scaled, y)
        
        # Predict and evaluate
        y_pred = model.predict(X_scaled)
        print("\nModel Performance:")
        print(classification_report(y, y_pred))
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        
        # Save model and scaler
        joblib.dump((model, scaler), model_file)
        print(f"Model saved to {model_file}")
    except Exception as e:
        print(f"Training error: {e}")
        print(traceback.format_exc())
        raise

# Predict Function
def predict(input_file, model_file, output_file):
    try:
        print(f"Loading model from {model_file}")
        model, scaler = joblib.load(model_file)
        
        print(f"Preparing input data from {input_file}")
        df = load_data(input_file)
        
        # Separate features (all columns except the last one)
        X = df.iloc[:, :-1]
        print(f"Input features shape: {X.shape}")
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        # Add predictions to original dataframe
        df['Predicted_Backorder'] = predictions
        
        # Save predictions
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Print prediction summary
        total_samples = len(predictions)
        failure_count = np.sum(predictions)
        print(f"\nPrediction Summary:")
        print(f"Total Samples: {total_samples}")
        print(f"Predicted Backorders: {failure_count}")
        print(f"Backorder Percentage: {(failure_count/total_samples)*100:.2f}%")
    
    except Exception as e:
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        raise

# CLI Setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backorder Prediction CLI")
    parser.add_argument("--train", help="Train MLP model using given CSV file")
    parser.add_argument("--predict", help="Predict backorders from a CSV file")
    parser.add_argument("--model", default="mlp_model.pkl", help="Path to model file")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
    args = parser.parse_args()
    
    try:
        if args.train:
            train_model(args.train, args.model)
        elif args.predict:
            predict(args.predict, args.model, args.output)
        else:
            print("Please provide either --train or --predict arguments")
    except Exception as e:
        print(f"CLI error: {e}")
        print(traceback.format_exc())
