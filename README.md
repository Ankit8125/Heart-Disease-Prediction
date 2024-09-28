# Heart Disease Prediction
This project aims to predict the presence of heart disease in a patient based on clinical parameters using various machine learning techniques. The notebook demonstrates the steps of loading data, exploratory data analysis (EDA), feature engineering, building machine learning models, and evaluating their performance.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Dependencies](#dependencies)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Modeling](#modeling)
7. [Evaluation](#evaluation)
8. [Scikit-Learn Workflow](#scikit-learn-workflow)

## Project Overview
The objective of this project is to predict heart disease using various medical attributes. The key workflow is as follows:

1. **Problem Definition**: 
   - Can we predict whether a patient has heart disease based on their clinical attributes?
   - This is a binary classification problem, where the presence or absence of heart disease is predicted.
2. **Data**: 
   - The dataset is sourced from the [Kaggle Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
   - It includes medical attributes such as age, sex, chest pain type, cholesterol levels, and more.
3. **Evaluation**: 
   - Maximizing the accuracy of predictions.
   - The best-performing model achieved **81.96%** accuracy using `train_test_split`.

## Dataset
The dataset contains the following features:

- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy (0–3)
- **thal**: Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)

## Installation
To run this project, you will need to install the following libraries:

```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-classification.git

# Navigate to the project directory
cd heart-disease-classification

# Install dependencies
pip install -r requirements.txt
```
## Dependencies:
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
  
## Exploratory Data Analysis (EDA)
The following steps were taken to analyze the data:
- **Data Cleaning**: Identified and handled missing values appropriately.
- **Visualization**: Explored data distributions, relationships between features, and correlations using Matplotlib and Seaborn.
- **Feature Analysis**: Investigated how key features (like age, chest pain type, and cholesterol) impact heart disease occurrence.

## Modeling
The following machine learning algorithms were implemented:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)** 
- **Random Forest Classifier** <br/>
Hyperparameters for these models were optimized using **RandomizedSearchCV** and **GridSearchCV** to improve accuracy.

## Scikit-Learn Workflow
1. Get data ready
2. Pick a model(to suit your problem)
3. Fit the model to the data and make a prediction
4. Evaluate the model
5. Improve through experimentation
6. Save and reload your trained model

## Evaluation
Model performance was assessed using the following metrics:
- **Confusion Matrix**: To analyze model predictions versus actual outcomes.
- **Precision, Recall, F1-Score**: To understand the balance between false positives and negatives.
- **ROC Curve**: To visualize the model’s ability to distinguish between classes.<br/>
The final model achieved an accuracy of **81.96%**, making it a reliable predictor of heart disease based on the provided features.
