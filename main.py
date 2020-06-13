from flask import Flask, request, render_template
import numpy as np
import pickle

# Use pickle to load in the pre-trained model.
with open(f'diabetes.pkl', 'rb') as f:
    model = pickle.load(f)

# file = open('diabetes', 'rb')
# clf = pickle.load(file)
# file.close()

app = Flask(__name__)

@app.route("/") #homepage
def home():
    return render_template('home.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('home.html'))
    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        age = request.form['age']
        input_variables = [[pregnancies, glucose, insulin, bmi, age]]
        prediction = model.predict(input_variables)[0]
        return render_template('home.html',
                                     original_input={'Pregnancies':pregnancies,
                                                     'Glucose':glucose,
                                                     'Insulin':insulin,
                                                     'BMI':bmi,
                                                     'Age':age},
                                     result=prediction,
                                     show=True
                                     )
if __name__ == "__main__":
    app.run(debug=True)