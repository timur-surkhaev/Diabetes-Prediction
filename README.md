# The Midterm Project for ML-Zoomcamp 2024

Predicting presence or absence of diabetes (including prediabetic conditions) based on a patient's health and demographic indicators

Data is from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

EDA, model training and selecting are provided in notebooks/diabetes_prediction.ipynb

The final process of model training, selecting and saving is provided in source/train.py



## Containerization

Docker image: https://hub.docker.com/repository/docker/tsurkhaev/diabetes-prediction/general

Pull the docker image: 
```bash
sudo docker pull tsurkhaev/diabetes-prediction:v1.0
```

Run the container: 
```bash
sudo docker run --rm -it -p 0.0.0.0:8000:8000 tsurkhaev/diabetes-prediction:v1.0
```

Optionally. Build the container locally (if you have a local copy of the repository; see below): 
```bash
sudo docker build -t diabetes-prediction:local .
```

## Virtual envinronment

Installing virtual envinronment:

1. Get the repository:
```bash
cd ~/local_path_to_the_project
git clone https://github.com/timur-surkhaev/Diabetes-Prediction.git
```

2. Install the virtual env:
```bash
pip install pipenv
pipenv install --ignore-pipfile
```

3. Activate the virtual envinronment (in the root project directory):
```bash
pipenv shell
```

## Test

After the docker-container has started, you might to send a test request (in the activated virtual envinronment):
```bash
python3 tests/test_request_01.py
```
