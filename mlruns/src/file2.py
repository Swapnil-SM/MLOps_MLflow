import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#code to connect to DagsHub the link and the code is copied in the remote->experiments->Mlflow UI option

import dagshub
dagshub.init(repo_owner='Swapnil-SM', repo_name='MLOps_MLflow', mlflow=True)

# import os
# os.environ['MLFLOW_TRACKING_USERNAME'] = "Swapnil-SM"
# os.environ['MLFLOW_TRACKING_PASSWORD'] = "6d8a1c6d2ec8251d9d0216a7c3a5c87de720d524"
mlflow.set_tracking_uri("https://dagshub.com/Swapnil-SM/MLOps_MLflow.mlflow")

mlflow.set_experiment('MLOPS-Exp2')

# import mlflow
# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)
# Load Wine dataset
from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 25
n_estimators = 25


#This starts tracking a new experiment run in MLflow.
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    #A metric is a value that measures how well your model is performing.
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")

    #__file__ → Refers to the current script's filename.
    #log_artifact() → Saves this file (artifact) in the MLflow run directory so you can track which code was used for that experiment.

    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Swapnil', "Project": "Wine Classification"})

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    # import time
    # model_path = f"Random-Forest-Model-{int(time.time())}"
    # mlflow.sklearn.save_model(rf, model_path)
    # mlflow.log_artifact(model_path)

    # mlflow.sklearn.save_model(rf, "Random-Forest-Model")
    # mlflow.log_artifact("Random-Forest-Model")


    print(accuracy)