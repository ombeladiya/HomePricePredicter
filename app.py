from flask import Flask, request, jsonify, render_template
import joblib
import json
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('home_price_model.pkl')
label_enc_mainroad = joblib.load('label_enc_mainroad.pkl')
label_enc_guestroom = joblib.load('label_enc_guestroom.pkl')
label_enc_basement = joblib.load('label_enc_basement.pkl')
label_enc_hotwaterheating = joblib.load('label_enc_hotwaterheating.pkl')
label_enc_airconditioning = joblib.load('label_enc_airconditioning.pkl')
label_enc_prefarea = joblib.load('label_enc_prefarea.pkl')
label_enc_furnishingstatus = joblib.load('label_enc_furnishingstatus.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Create DataFrame from input data
    input_df = pd.DataFrame([data])
   
    # Encode categorical features
    input_df['mainroad'] = label_enc_mainroad.transform(input_df['mainroad'])
    input_df['guestroom'] = label_enc_guestroom.transform(input_df['guestroom'])
    input_df['basement'] = label_enc_basement.transform(input_df['basement'])
    input_df['hotwaterheating'] = label_enc_hotwaterheating.transform(input_df['hotwaterheating'])
    input_df['airconditioning'] = label_enc_airconditioning.transform(input_df['airconditioning'])
    input_df['prefarea'] = label_enc_prefarea.transform(input_df['prefarea'])
    input_df['furnishingstatus'] = label_enc_furnishingstatus.transform(input_df['furnishingstatus'])

    # Ensure the feature columns are in the same order as the training data
    features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
    input_df = input_df[features]

    # Predict price
    prediction = model.predict(input_df)[0]

    # Convert prediction to integer
    predicted_price = int(prediction)
    return jsonify({'predicted_price': predicted_price})

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
