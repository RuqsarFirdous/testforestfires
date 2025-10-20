import pickle
import numpy as np
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('ridge.pkl','rb'))
standard_scaler = pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        # Get inputs from form
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        # Scale and predict
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)[0]

        # Render the home page with the prediction
        return render_template('home.html', result=result)

    else:
        # GET request → just show home.html without any prediction
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
