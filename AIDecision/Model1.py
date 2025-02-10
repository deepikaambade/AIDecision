import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import click
import joblib

class BackorderPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None

    def load_and_analyze_dataset(self, filepath):
        """
        Load and perform initial dataset analysis
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file containing backorder data
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataset
        """
        # Load dataset
        df = pd.read_csv(filepath)
        
        # Basic dataset information
        print("Dataset Information:")
        print(df.info())
        
        # Missing values analysis
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Descriptive statistics
        print("\nDescriptive Statistics:")
        print(df.describe())
        
        # Distribution of target variable (if exists)
        if 'backorder' in df.columns:
            plt.figure(figsize=(8, 6))
            df['backorder'].value_counts().plot(kind='bar')
            plt.title('Distribution of Backorder')
            plt.xlabel('Backorder Status')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('backorder_distribution.png')
            plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        return df

    def preprocess_data(self, df, target_column='backorder'):
        """
        Preprocess the dataset for MLP training
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_column : str, optional
            Name of the target column
        
        Returns:
        --------
        tuple
            Processed features (X) and target variable (y)
        """
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target variable if not binary
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded

    def train_mlp_model(self, X, y, hidden_layers=(50, 25), max_iter=500):
        """
        Train Multi-Layer Perceptron Classifier
        
        Parameters:
        -----------
        X : np.array
            Scaled feature matrix
        y : np.array
            Encoded target variable
        hidden_layers : tuple, optional
            Configuration of hidden layers
        max_iter : int, optional
            Maximum number of iterations
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train MLP
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers, 
            max_iter=max_iter,
            activation='relu',
            solver='adam',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def save_model(self, filepath='backorder_model.pkl'):
        """
        Save trained model and preprocessing objects
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the model
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='backorder_model.pkl'):
        """
        Load pre-trained model and preprocessing objects
        
        Parameters:
        -----------
        filepath : str, optional
            Path to load the model from
        """
        saved_objects = joblib.load(filepath)
        self.model = saved_objects['model']
        self.scaler = saved_objects['scaler']
        self.label_encoder = saved_objects['label_encoder']
        print("Model loaded successfully")

    def predict(self, X_new):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            New data for prediction
        
        Returns:
        --------
        np.array
            Predicted labels
        """
        # Preprocess new data
        X_scaled = self.scaler.transform(X_new)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Decode predictions
        return self.label_encoder.inverse_transform(predictions)

@click.group()
def cli():
    """Backorder Prediction CLI"""
    pass

@cli.command()
@click.option('--dataset', '-d', required=True, help='Path to the dataset CSV')
def analyze(dataset):
    """Analyze the dataset and generate visualizations"""
    predictor = BackorderPredictor()
    df = predictor.load_and_analyze_dataset(dataset)
    click.echo("Dataset analysis complete. Check generated visualizations.")

@cli.command()
@click.option('--dataset', '-d', required=True, help='Path to the dataset CSV')
def train(dataset):
    """Train the MLP model on the given dataset"""
    predictor = BackorderPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_analyze_dataset(dataset)
    X, y = predictor.preprocess_data(df)
    
    # Train model
    predictor.train_mlp_model(X, y)
    
    # Save model
    predictor.save_model()
    click.echo("Model training complete.")

@cli.command()
@click.option('--input', '-i', required=True, help='Input CSV for predictions')
@click.option('--output', '-o', default='predictions.csv', help='Output CSV for predictions')
def predict_cmd(input, output):
    """Make predictions on new data"""
    predictor = BackorderPredictor()
    
    # Load model
    predictor.load_model()
    
    # Load input data
    input_data = pd.read_csv(input)
    
    # Predict
    predictions = predictor.predict(input_data)
    
    # Save predictions
    input_data['predicted_backorder'] = predictions
    input_data.to_csv(output, index=False)
    click.echo(f"Predictions saved to {output}")

if __name__ == '__main__':
    cli()
