FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "the_best_model.pkl", "./"]

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "predict:app"]
