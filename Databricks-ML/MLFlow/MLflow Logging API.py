# Databricks notebook source
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from numpy import savetxt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# COMMAND ----------

db = load_diabetes()

X = db.data
y = db.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
# The code above loads the diabetes dataset from sklearn's datasets module.
# 'db' is a dictionary-like object containing the dataset.
# 'X' is assigned the feature data from the dataset.
# 'y' is assigned the target values from the dataset.

# COMMAND ----------

print("Number of observations in X_train are:" ,len(X_train))
print("Number of observations in X_test are:" ,len(X_test))
print("Number of observations in y_train are:" ,len(y_train))
print("Number of observations in y_test are:" ,len(y_test))

# COMMAND ----------

n_estimators = 100
max_depth = 6
max_features = 3

# Create and train model
rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset
predictions = rf.predict(X_test)

# COMMAND ----------

# Enable autolog()
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and fitted model are automatically logged
with mlflow.start_run():
    # Set the model parameters
    n_estimators = 120
    max_depth = 6
    max_features = 3

    # Create and train model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset
    predictions = rf.predict(X_test)

# COMMAND ----------

experiment_name = "/Shared/diabetes_experiment"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    # Set the model parameters
    n_estimators = 100
    max_depth = 6
    max_features = 3

    # Create and train model
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset
    predictions = rf.predict(X_test)

    # Log parameters
    mlflow.log_param("nun_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_features", max_features)

    # Log model
    mlflow.sklearn.log_model(rf, "random_forest_model")

    # Create metrics
    mse = mean_squared_error(y_test, predictions)


# COMMAND ----------


