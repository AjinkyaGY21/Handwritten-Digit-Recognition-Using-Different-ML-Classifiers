# Handwritten Digit Recognition Project 🔢🤖

## Project Overview
This project implements a machine learning pipeline for recognizing handwritten digits using multiple classification algorithms. The goal is to classify handwritten digits from the MNIST-like dataset with high accuracy.

## Features
- **Multiple Machine Learning Classifiers:** 
  - 🧠 Support Vector Machine (SVM)
  - 🌲 Decision Tree
  - 🌳 Random Forest
  - 📊 Naive Bayes
  - 🧮 Multi-Layer Perceptron (MLP)
  - 📈 XGBoost
  - 🚀 Gradient Boosting

- **Data Preprocessing:** 
  - 📏 Feature scaling using StandardScaler
  - 🔀 Train-validation split
  - 📝 Performance evaluation metrics

- **Visualization:** 
  - 🗂️ Confusion matrix for each classifier
  - 📊 Accuracy comparison bar plot
  - 💾 Save results and predictions to CSV files

## Prerequisites
- 🐍 Python 3.8+
- Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - XGBoost

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## Dataset
- 📊 Training dataset: `train.csv`
- 📋 Test dataset: `test.csv`
- 🔗 Download from: "https://www.kaggle.com/competitions/digit-recognizer/data"

## Key Outputs
- `classification_results.csv`: Model performance metrics
- `test_predictions.csv`: Predictions from different models and majority vote

## Performance Metrics
- 📈 Train and validation accuracy for each model
- 🗂️ Confusion matrices
- 🤝 Majority voting for final predictions on test dataset

## Customization
- Modify `classifiers` dictionary to add/remove models
- Adjust hyperparameters for fine-tuning
- Change train-test split ratio

## Visualizations
- 📊 Bar plot comparing train and validation accuracies
- 🗂️ Confusion matrix for each classifier
