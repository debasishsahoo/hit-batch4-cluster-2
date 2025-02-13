from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Sample training data (Simple Linear Regression)
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([10, 20, 30, 40, 50])

# Train and save the model
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_input = np.array(data['features']).reshape(-1, 1)
        model = joblib.load("model.pkl")
        prediction = model.predict(X_input).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
