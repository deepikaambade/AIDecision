# AI Decision: Machine Failure Prediction System

## 🚀 Project Overview
The **AI Decision: Machine Failure Prediction System** is a Machine Learning (ML) Command-Line Interface (CLI) application that predicts machine failures using a **Multi-Layer Perceptron (MLP) classifier**. The system is designed for **easy deployment, scalability, and reproducibility**.

## 📂 Project Structure
```
AIDecision/
│
├── ai4i2020.csv          # Original dataset
├── cli.py                # Main CLI application
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── model_final.ipynb     # Jupyter notebook with model exploration
├── requirements.txt      # Python dependencies
│
├── pkl_files/            # Serialized data and model
│   ├── X_train.pkl       # Training features
│   ├── X_test.pkl        # Testing features
│   ├── y_train.pkl       # Training labels
│   ├── y_test.pkl        # Testing labels
│   └── mlp_model.pkl     # Trained MLP model
│
├── templates.html        # Web UI template
├── model2.py             # Alternative model implementation
├── app.py                # Flask web application
├── ai4i2020.xls          # Excel version of the dataset
└── predictions.csv        # Model predictions output
```

---

## 🤖 MLP Model Architecture

### Model Configuration:
- **Type:** Multi-Layer Perceptron (Neural Network Classifier)
- **Hidden Layers:**
  - First Layer: 50 neurons
  - Second Layer: 25 neurons
- **Activation Function:** ReLU

### Training Parameters:
- **Max Iterations:** 500
- **Random State:** 42

### Data Preprocessing:
- Categorical variables are **label-encoded**
- Numerical features are **standardized**
- Missing values are **handled through mean imputation**

---

## 📊 Dataset Insights
- **Source:** ai4i2020.csv
- **Total Samples:** 10,000
- **Features:**
  - Air Temperature
  - Process Temperature
  - Rotational Speed
  - Torque
  - Tool Wear
- **Target Variable:** Machine Failure (**Binary Classification**)

---

## 🔍 Jupyter Notebook Exploration
The **model_final.ipynb** provides:
- Detailed **data exploration**
- **Feature engineering techniques**
- **Model selection** process
- **Hyperparameter tuning**
- **Visualization** of model performance

---

## 🚧 Future Improvements
- Implement **advanced feature selection**
- Explore **ensemble methods**
- Add **more robust error handling**
- Create **comprehensive logging**

---

## 🐍 Python Scripts

### 1️⃣ cli.py
- Primary command-line interface for ML operations
- Handles **model training, prediction, and evaluation**
- Supports **data loading, preprocessing, and performance metrics**
- Provides **command-line arguments** for flexible execution
- Generates **visualizations** (confusion matrix, feature importance)

### 2️⃣ model2.py
- **Alternative or experimental** model implementation
- May include **additional modeling approaches or variations**

### 3️⃣ app.py
- **Flask web application**
- Provides **a web-based interface** for predictions
- Handles **file uploads and prediction requests**
- Serves as **the backend** for the web-based prediction system

---

## 📊 Data Files

### 1️⃣ ai4i2020.csv
- **Original dataset** for training and testing the ML model
- Includes features like **air temperature, process temperature, rotational speed, etc.**

### 2️⃣ ai4i2020.xls
- **Excel version** of the same dataset
- Useful for **manual data exploration**

---

## 🧊 Pickle Files

### 1️⃣ X_train.pkl
- **Preprocessed training features** (NumPy array or DataFrame)
- Speeds up model training and ensures consistent preprocessing

### 2️⃣ X_test.pkl
- **Preprocessed testing features** for model evaluation

### 3️⃣ y_train.pkl
- **Training labels** (binary classification target)

### 4️⃣ y_test.pkl
- **Testing labels** for model evaluation

### 5️⃣ mlp_model.pkl
- **Serialized trained MLP model**
- Allows quick model loading without retraining

---

## 📓 Jupyter Notebook: model_final.ipynb
- Comprehensive **model exploration**
- Contains **data preprocessing steps**
- **Model training and tuning**
- **Visualization of results**
- **Experimental analysis**

---

## 🌐 Web Interface: templates.html
- **HTML template** for the web UI
- Provides **file upload and prediction interface**
- Includes **client-side JavaScript for interaction**

---

## 🐳 Containerization

### 1️⃣ Dockerfile
- Defines **Docker container configuration**
- Specifies **Python environment**
- Sets up **dependencies and application launch**

### 2️⃣ docker-compose.yml
- **Docker Compose configuration**
- Defines **services, ports, and environment variables**
- Simplifies **container management**

---

## 📦 Dependency Management: requirements.txt
- Lists all **Python package dependencies**
- Ensures **consistent environment setup**
- Includes **libraries for data science, ML, and web development**

---

## 📄 Documentation: README.md
- **Project documentation**
- Provides **setup instructions**
- Explains **project structure and usage**
- Offers insights into **the machine learning approach**

---

## 📊 Output: predictions.csv
- CSV file containing **model predictions**
- Stores **results of machine failure predictions**
- Can be used for **further analysis or reporting**