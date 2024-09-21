# Heart Disease Prediction
This project aims to predict the presence of heart disease in a patient based on clinical parameters using various machine learning techniques. The notebook walks through the steps of loading data, exploratory data analysis (EDA), building models, and evaluating their performance.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)

## Project Overview
The objective of this project is to use machine learning algorithms to predict heart disease based on various medical attributes. The following approach is used:

1. **Problem Definition**: 
   - Given clinical parameters about a patient, can we predict whether or not they have heart disease?
2. **Data**: 
   - Sourced from the Kaggle Heart Disease dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
3. **Evaluation**: 
   - Maximizing the accuracy of predictions.
   - Accuracy achieved: **81.96%**

## Dataset
The dataset contains the following features:

- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type
  - 0: Typical angina
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic
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
The notebook explores the following aspects of the data:
- Data distribution and missing values.
- Relationships between features and the target variable.
- Visualizations to better understand the dataset using matplotlib and seaborn.

## Modeling
The following machine learning algorithms are used:
- Logistic Regression <br>
- K-Nearest Neighbors (KNN) <br>
- Random Forest Classifier <br><br>
Model hyperparameters are tuned using RandomizedSearchCV and GridSearchCV to find the optimal configuration.

## Evaluation
Model performance is evaluated based on:
- Confusion Matrix, Precision, Recall, F1-Score. ROC Curve
- The final model achieved an accuracy of 81.96%.
