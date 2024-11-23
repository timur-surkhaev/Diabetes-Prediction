import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import auc, roc_curve, roc_auc_score, f1_score, precision_recall_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print(pd.__version__)


# Preparing data
df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
df.columns = ['Diabetes'] + list(df.columns[1:]) # Target variable

print(df.shape)

y = 'Diabetes'

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

X_train = df_train.loc[:, df.columns != y].copy()
X_val = df_val.loc[:, df.columns != y].copy()
X_test = df_test.loc[:, df.columns != y].copy()

y_train = df_train[y].values
y_val = df_val[y].values
y_test = df_test[y].values

for d in [X_train, X_val, X_test]:
    print(round(len(d)/len(df),2))
    
# Training
model_hyperparams = {
        'LogisticRegression': {
            'model': LogisticRegression(solver='liblinear', max_iter=1000, random_state=1),
            'params': {
                'C': [0.01, 0.1, 1, 6, 6.5, 7, 7.5, 8, 9, 10, 50, 100],
                'penalty': ['l1', 'l2']
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(random_state=1),
            'params': {
                'max_depth': [7, 8, 9, 10, 11],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 4, 8]
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=1),
            'params': {
                'n_estimators': [5, 10, 15, 20],
                'max_depth': [7, 8, 9, 10, 11],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 4, 8]
            }
        }
}

best_models = {}

for model_name, mp in model_hyperparams.items():
    print(f"Running GridSearchCV for {model_name}...")
    grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='f1', n_jobs=4)
    
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = [grid_search.best_estimator_, grid_search.best_score_]
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation score: {round(grid_search.best_score_, 3)}\n")

for model_name, model in best_models.items():
    print(f"\nEvaluating the best {model_name} on the validation set:")
    y_pred = model[0].predict(X_val)
    print(classification_report(y_val, y_pred))

the_best_model = best_models[max(best_models, key=lambda k: best_models[k][1])][0]

y_pred = the_best_model.predict(X_val)
y_pred_proba = the_best_model.predict_proba(X_val)[:, 1]

fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

auc_value = auc(fpr, tpr)
f1_value = f1_score(y_val, y_pred)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10) # 1e-10 is here to avoid division by zero
best_threshold = thresholds[np.argmax(f1_scores)]

y_pred_thr = [1 if y > best_threshold else 0 for y in y_pred_proba]
f1_thr = f1_score(y_val, y_pred_thr)

print(f'AUC:\t{round(auc_value, 3)}')
print(f'F1:\t{round(f1_value, 3)}')
print(f'F1 thr:\t{round(f1_thr, 3)}')
print(f"Best threshold (F1-Score): {round(best_threshold, 3)}")

# Saving the best model
with open('models/the_best_model.pkl', 'wb') as file:
    pickle.dump(the_best_model, file)
