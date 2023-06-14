import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from prefect import task, flow
import os
import joblib




@task(
        name = "read data",
        retries = 3,
        retry_delay_seconds= 5
        )
def load_data(insurance_dataset=r'C:/Users/User/Documents/medicalinsurance/venv/data.csv'):

    # path to the insurance.csv file
    # loading the data from csv file to a pandas dataframe
    insurance_dataset = pd.read_csv(insurance_dataset)
    return insurance_dataset

@task(
        name = "explolatory data analysis",
        retries = 3,
        retry_delay_seconds= 5
)
def data_analysis(insurance_dataset):
    # Display basic information about the dataset
    print("First 5 rows of the dataframe:")
    print(insurance_dataset.head())

    print("Number of rows and columns:")
    print(insurance_dataset.shape)

    print("Information about the dataset:")
    print(insurance_dataset.info())

    print("Checking for missing values:")
    print(insurance_dataset.isnull().sum())

    # ... Perform data analysis and visualization as before ...

    # Return the preprocessed dataset
    return insurance_dataset

@task(
        name = "preprocess data",
        retries = 3,
        retry_delay_seconds= 5
)
def preprocess_data(insurance_dataset):
    # Encoding categorical features
    # Encoding categorical features
    insurance_dataset['sex'] = insurance_dataset['sex'].map({'male': 0, 'female': 1})
    insurance_dataset['smoker'] = insurance_dataset['smoker'].map({'no': 0, 'yes': 1})
    # ... Perform encoding of sex, smoker, and region columns ...
    # One-hot encoding for 'region' column
    encoded_regions = pd.get_dummies(insurance_dataset['region'], prefix='region')
    insurance_dataset = pd.concat([insurance_dataset, encoded_regions], axis=1)
    insurance_dataset.drop(columns=['region'], inplace=True)
    # Splitting features and target variables
    X = insurance_dataset.drop(columns='charges', axis=1)
    Y = insurance_dataset['charges']

    return X, Y

@task(
        name = "split data",
        retries = 3,
        retry_delay_seconds= 5
)
def split_data(X, Y, test_size):
    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=2)
    return X_train, X_test, Y_train, Y_test

@task(
        name = "train model",
        retries = 3,
        retry_delay_seconds= 5
)
def train_model(X_train, Y_train):
    # Training the linear regression model
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    return regressor

@task(
        name = "evaluate model",
        retries = 3,
        retry_delay_seconds= 5
)
def evaluate_model(regressor, X_train, Y_train, X_test, Y_test):
    # Evaluating the model
    training_data_prediction = regressor.predict(X_train)
    r2_train = metrics.r2_score(Y_train, training_data_prediction)
    print('R squared value (Training):', r2_train)

    test_data_prediction = regressor.predict(X_test)
    r2_test = metrics.r2_score(Y_test, test_data_prediction)
    print('R squared value (Testing):', r2_test)

    return r2_train, r2_test

@task(
        name = "predict data",
        retries = 3,
        retry_delay_seconds= 5
)
def predict_cost(regressor, age, sex, bmi, children, smoker, region):
    # Making predictions
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = regressor.predict(input_data)
    print('The insurance cost is USD', prediction[0])
    return prediction

# Define the workflow logic function
@flow
def run_workflow():
    data = load_data()
    analysis = data_analysis(data)
    X, Y = preprocess_data(analysis)
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)
    model = train_model(X_train, Y_train)
    r2_train, r2_test = evaluate_model(model, X_train, Y_train, X_test, Y_test)
    prediction = predict_cost(
        model,
        age=input('Enter age: '),
        sex=input('Enter sex: '),
        bmi=input('Enter BMI: '),
        children=input('Enter number of children: '),
        smoker=input('Enter smoker status (yes/no): '),
        region=input('Enter region: ')
    )



if __name__ == "__main__":
   run_workflow()


# Save the trained model
#joblib.dump(model, "model.pkl")