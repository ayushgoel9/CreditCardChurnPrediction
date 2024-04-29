from flask import Flask, request, render_template
import pandas as pd
from neural_net import model  # Import your prediction function from your model script
import joblib

# Load the model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_from_excel', methods=['POST'])
def predict_from_excel():
    # Get the uploaded file
    uploaded_file = request.files['file']

    # Read data from the Excel file
    data = pd.read_excel(uploaded_file)
    

    # Pass data to the prediction function
    prediction = model.predict(data)  # Pass the entire DataFrame to the prediction function

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

