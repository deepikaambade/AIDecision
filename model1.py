import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

# Function to load CSV data
def load_data(csv_path):
    """
    Loads the dataset from a CSV file and prints basic details.
    Justification: Helps in understanding the dataset structure before processing.
    """
    data = pd.read_csv(csv_path)
    print("Dataset Loaded Successfully!")
    print("Columns:", list(data.columns))
    print("Number of Rows:", data.shape[0])
    print("Number of Columns:", data.shape[1])
    return data

# Function for EDA and visualization
def exploratory_data_analysis(data):
    """
    Performs Exploratory Data Analysis (EDA) including summary statistics,
    missing values check, and visualizations.
    Justification: Helps understand feature distributions and correlations for feature selection.
    """
    print("\nBasic Statistics:\n", data.describe())
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nFirst 5 Rows:\n", data.head())
    
    numerical_cols = data.select_dtypes(include=['number']).columns
    data[numerical_cols].hist(figsize=(12, 10))  # Visualizing data distributions
    plt.show()
    
    # Correlation heatmap to detect multicollinearity
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.show()
    
    # Asking user for custom scatter plot visualization
    print("Available columns:", list(data.columns))
    x_col = input("Enter the column number to use for X-axis: ")
    y_col = input("Enter the column number to use for Y-axis: ")
    
    try:
        x_col = int(x_col)
        y_col = int(y_col)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=data.columns[x_col], y=data.columns[y_col])
        plt.show()
    except:
        print("Invalid column index. Skipping scatter plot.")

# Function to preprocess data
def preprocess_data(data, target_column):
    """
    Preprocesses data by handling missing values, feature selection, and scaling.
    Justification: Ensures model receives clean and standardized data.
    """
    exploratory_data_analysis(data)
    
    data = data.dropna()  # Removing rows with missing values
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X = X.loc[:, X.var() > 0.01]  # Removing low-variance features
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# PyTorch-based MLP Model Class
def train_torch_mlp(X_train, y_train, input_size, hidden_size=64, num_epochs=50, learning_rate=0.01):
    """
    Defines and trains a Multi-Layer Perceptron (MLP) classifier using PyTorch.
    Justification: PyTorch allows flexible deep learning model training.
    """
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    model = MLP(input_size, hidden_size)
    criterion = nn.BCELoss()  # Binary Cross Entropy for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'mlp_torch.pth')
    return model

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    """
    Evaluates the model using classification metrics and a confusion matrix.
    Justification: Helps in understanding model effectiveness and areas of improvement.
    """
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Accuracy Score:", accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main CLI function
def main():
    parser = argparse.ArgumentParser(description='Backorder Prediction CLI')
    parser.add_argument('csv_file', type=str, help='Path to input CSV file')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', action='store_true', help='Predict using trained model')
    args = parser.parse_args()

    data = load_data(args.csv_file)
    target_column = 'backorder'  # Change this to match your dataset
    X, y, scaler = preprocess_data(data, target_column)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if args.train:
        model = train_torch_mlp(X_train, y_train, input_size=X_train.shape[1])
        y_pred = (model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy() > 0.5).astype(int)
        evaluate_model(y_test, y_pred)
    
    if args.predict:
        model = joblib.load('mlp_model.pkl')
        predictions = predict_backorder(model, X)
        save_predictions(predictions, 'predictions.csv')
        print("Predictions saved to predictions.csv")

if __name__ == '__main__':
    main()
