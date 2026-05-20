from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

df = pd.read_csv("carData.csv")

@app.route('/')
def home():
    companies = sorted(df['company'].unique())
    names = sorted(df['name'].unique())
    fuels = sorted(df['fuel_type'].unique())

    return render_template("index.html",
                           companies=companies,
                           names=names,
                           fuels=fuels)

@app.route('/predict', methods=['POST'])
def predict():

    year = int(request.form['year'])
    kms = int(request.form['kms_driven'])
    company = request.form['company']
    name = request.form['name']
    fuel = request.form['fuel_type']

    # create input dataframe
    input_df = pd.DataFrame([[year, kms, company, name, fuel]],
                            columns=['year','kms_driven','company','name','fuel_type'])

    input_df = pd.get_dummies(input_df)

    # align columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    return render_template("index.html",
                           prediction_text=f"🚗 Estimated Price: {round(prediction,2)}")

if __name__ == "__main__":
    app.run(debug=True)