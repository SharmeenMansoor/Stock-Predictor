from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Upload CSV and predict next day's price
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Read CSV (expecting columns: Date, Close)
    df = pd.read_csv(filepath)
    if 'Close' not in df.columns:
        return jsonify({'error': 'CSV must contain a "Close" column'}), 400

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(inplace=True)

    # Prepare data for prediction
    df['Day'] = np.arange(len(df))
    X = df['Day'].values.reshape(-1, 1)
    y = df['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[len(df)]])
    prediction = model.predict(next_day)[0]

    return jsonify({'prediction': round(prediction, 2)})


if __name__ == '__main__':
    app.run(debug=True)
