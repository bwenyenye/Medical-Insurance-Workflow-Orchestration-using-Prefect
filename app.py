# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

app = Flask(__name__)

# Load the pre-trained model
model = LinearRegression()
model.load('model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/prediction', methods=['POST'])
def prediction():
    # Get the form data
    data = request.get_json()
    age = float(data['age'])
    sex = float(data['sex'])
    bmi = float(data['bmi'])
    children = float(data['children'])
    smoker = float(data['smoker'])
    region = float(data['region'])

    # Make the prediction
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)[0]

    # Return the prediction result
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
