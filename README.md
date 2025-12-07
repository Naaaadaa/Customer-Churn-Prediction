# Customer Churn Prediction (Telecom Industry)

This project focuses on predicting whether a telecom customer is likely to churn using the Telco Customer Churn dataset from Kaggle. Our group worked collaboratively through an end-to-end machine learning pipeline, including data cleaning, exploratory analysis, model building, evaluation, and deployment. The aim was to understand the key factors that influence churn and to develop a model that can accurately identify customers who are at risk of leaving the service.

---

## Project Overview

Customer churn is a major challenge for telecom companies because losing customers directly impacts revenue. By predicting churn in advance, companies can take steps to retain high-risk customers.  
In this project, the team explored the dataset, engineered features, trained multiple machine-learning models, evaluated their performance, and deployed the best model using Streamlit Cloud.

---

## Dataset

The dataset used is the **Telco Customer Churn** dataset from Kaggle. It contains **7032 customer records** and a mix of numerical and categorical variables. The target variable is **Churn** (Yes/No).

Key features include:

- Customer account details (tenure, contract type, payment method)  
- Billing information (MonthlyCharges, TotalCharges)  
- Services subscribed (internet, phone, security options)  
- Demographic information  

---

## Data Cleaning and Preprocessing

As a group, we performed the following steps to ensure data quality:

- Converted `TotalCharges` from text to numeric and removed invalid rows  
- Checked and handled missing values  
- Applied one-hot encoding to categorical variables  
- Created new features to support learning  
- Split the data into training and testing sets  

These steps prepared the dataset for accurate and consistent model training.

---

## Exploratory Data Analysis

To understand patterns in the data, the group analysed relationships between features and churn.  
Key findings include:

- Customers with **month-to-month contracts** churned at much higher rates  
- Shorter tenure customers were more likely to leave  
- Higher monthly charges were associated with churn  
- Certain services (such as tech support) showed strong links to churn behaviour  

Visualisations such as histograms, boxplots, and correlation charts supported these observations.

---

## Models Trained

The team trained and compared three machine-learning models:

1. **Logistic Regression**  
2. **Random Forest Classifier**  
3. **Gradient Boosting Classifier**

Each model was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## Model Performance Summary

Among all models tested, **Logistic Regression** provided the strongest overall performance. It achieved the best balance between recall and F1-score, making it the most effective at identifying true churners. Random Forest and Gradient Boosting performed reasonably well but did not surpass Logistic Regression for this dataset.

---

## Deployment

To demonstrate how the model can be used in practice, the team deployed the final model using **Streamlit Cloud**.  
The web app allows users to input customer details and receive an instant churn prediction along with the probability score.

Deployment steps included:

- Saving the trained model and preprocessing pipeline  
- Creating a Streamlit script (`streamlit_app.py`) to handle inputs and predictions  
- Uploading all files to GitHub  
- Deploying the app through Streamlit Cloud  

This deployment shows how machine learning models can b

