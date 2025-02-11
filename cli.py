import argparse
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

class BackorderPredictor:
    def __init__(self, model_path='mlp_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None

    def preprocess_data(self, df):
        """
        Comprehensive preprocessing of the input dataframe
        
        :param df: Input DataFrame
        :return: Preprocessed features
        """
        # Create a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Replace '?' with NaN
        df.replace("?", np.nan, inplace=True)
        
        # Columns to drop
        columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type']
        
        # Drop unnecessary columns
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
        
        # Identify and handle columns
        for col in df.columns:
            # If column is object type
            if df[col].dtype == 'object':
                # Try to extract numeric part if possible
                try:
                    # Remove any non-numeric prefix/suffix and convert to float
                    df[col] = df[col].str.extract('(\d+\.?\d*)')[0].astype(float)
                except:
                    # If extraction fails, drop the column
                    print(f"Dropping column {col} due to non-numeric content")
                    df.drop(columns=[col], axis=1, inplace=True)
        
        # Ensure all remaining columns are numeric
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_columns]
        
        # Fill missing values with mean
        df.fillna(df.mean(), inplace=True)
        
        return df

    def load_default_data(self):
        """
        Load default training data from pickle files
        
        :return: X_train, y_train, X_test, y_test
        """
        try:
            with open("X_train.pkl", "rb") as f:
                X_train = pickle.load(f)
            with open("y_train.pkl", "rb") as f:
                y_train = pickle.load(f)
            with open("X_test.pkl", "rb") as f:
                X_test = pickle.load(f)
            with open("y_test.pkl", "rb") as f:
                y_test = pickle.load(f)
            return X_train, y_train, X_test, y_test
        except FileNotFoundError:
            # Fallback to CSV if pickle files are not found
            print("Pickle files not found. Attempting to load from CSV.")
            df = pd.read_csv('ai4i2020.csv')
            
            # Separate features and target
            X = self.preprocess_data(df.drop('Machine failure', axis=1))
            y = df['Machine failure']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, y_train, X_test, y_test

    def train_model(self):
        """
        Train the MLP model
        """
        # Load data
        X_train, y_train, X_test, y_test = self.load_default_data()
        
        # Scale the features
        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train MLP
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # Two hidden layers
            activation='relu',
            max_iter=500,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # Save the model and scaler
        with open(self.model_path, "wb") as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"Model saved to {self.model_path}")

    def predict(self, csv_path):
        """
        Predict backorders for a given CSV file
        
        :param csv_path: Path to input CSV file
        """
        # Load the trained model
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
        except FileNotFoundError:
            print(f"No trained model found at {self.model_path}. Please train the model first.")
            return
        
        # Read and preprocess input data
        df = pd.read_csv(csv_path)
        X = self.preprocess_data(df)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to original dataframe
        df['Predicted_Backorder'] = predictions
        df['Backorder_Probability'] = probabilities
        
        # Save predictions
        output_file = "predictions.csv"
        df.to_csv(output_file, index=False)
        
        # Print summary
        total_samples = len(predictions)
        backorder_count = np.sum(predictions)
        print(f"\nPrediction Summary:")
        print(f"Total Samples: {total_samples}")
        print(f"Predicted Backorders: {backorder_count}")
        print(f"Backorder Percentage: {(backorder_count/total_samples)*100:.2f}%")
        print(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Backorder Prediction CLI")
    parser.add_argument("--train", action="store_true", help="Train the MLP model")
    parser.add_argument("--predict", type=str, help="Path to CSV file for prediction")
    args = parser.parse_args()
    
    predictor = BackorderPredictor()
    
    if args.train:
        predictor.train_model()
    elif args.predict:
        predictor.predict(args.predict)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
