from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Use pickle to load in the pre-trained model.
with open(f'diabetes.pkl', 'rb') as f:
    model = pickle.load(f)

# file = open('diabetes', 'rb')
# clf = pickle.load(file)
# file.close()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/getdelay', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('home.html'))
    if request.method == 'POST':
        pregnancies = request.form['Pregnancies']
        glucose = request.form['Glucose']
        insulin = request.form['Insulin']
        bmi = request.form['BMI']
        age = request.form['Age']
        input_variables = pd.DataFrame([[pregnancies, glucose, insulin, bmi, age]],
                                       columns=['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return render_template('home.html',
                                     original_input={'Pregnancies':pregnancies,
                                                     'Glucose':glucose,
                                                     'Insulin':insulin,
                                                     'BMI':bmi,
                                                     'Age':age},
                                     result=prediction,
                                     )
if __name__ == "__main__":
    app.run(debug=True)