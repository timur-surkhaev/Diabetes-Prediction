# python3.11 predict.py


import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'models/the_best_model.pkl'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
    

app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    patient = np.array(list(patient.values())).reshape(1,-1)
    
    y_proba = model.predict_proba(patient)[0, 1]
    y_bin = y_proba > 0.192

    result = {
        'y_proba': float(y_proba),
        'y_bin': bool(y_bin)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
