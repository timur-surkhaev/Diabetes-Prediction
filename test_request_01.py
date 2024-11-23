# python3.11 test_request_01.py

import requests

url = 'http://127.0.0.1:8000/predict'


client = {"job": "student", "duration": 280, "poutcome": "failure"}

patient = {
            "HighBP":1.0,
            "HighChol":0.0,
            "CholCheck":1.0,
            "BMI":42.0,
            "Smoker":1.0,
            "Stroke":0.0,
            "HeartDiseaseorAttack":0.0,
            "PhysActivity":1.0,
            "Fruits":1.0,
            "Veggies":1.0,
            "HvyAlcoholConsump":0.0,
            "AnyHealthcare":1.0,
            "NoDocbcCost":0.0,
            "GenHlth":5.0,
            "MentHlth":6.0,
            "PhysHlth":30.0,
            "DiffWalk":1.0,
            "Sex":0.0,
            "Age":7.0,
            "Education":6.0,
            "Income":1.0
            }

response = requests.post(url, json=patient).json()

print(response)
