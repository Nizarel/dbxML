# Databricks notebook source
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

# COMMAND ----------

import pandas as pd
import mlflow
from pyspark.sql.functions import monotonically_increasing_id, expr, rand
import uuid

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

raw_data = spark.read.load("/databricks-datasets/wine-quality/winequality-red.csv", format="csv", sep = ";", header="true", inferSchema="true")

# COMMAND ----------

def addIdColumn(dataframe, id_column_name):
    columns= dataframe.columns
    new_df = dataframe.withColumn(id_column_name, monotonically_increasing_id())
    return new_df[[id_column_name] + columns]

def renameColumns(df):
    renamed_df = df
    for column in df.columns:
        renamed_df = renamed_df.withColumnRenamed(column, column.replace(' ', '_'))
    return renamed_df

# COMMAND ----------

renamed_df = renameColumns(raw_data)
df = addIdColumn(renamed_df, "wine_id")

# COMMAND ----------

features_df = df.drop('quality')
display(features_df)

# COMMAND ----------

print(features_df.schema)

# COMMAND ----------

spark.sql(f"CREATE TABLE IF NOT EXISTS wine_db")

table_name = f"wine_db_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["wine_id"],
    df=features_df,
    schema=features_df.schema,
    description="Wine Features"
)

# COMMAND ----------

inference_data_df = df.select("wine_id", "quality", (10*rand()).alias("real_time_measurement"))

display(inference_data_df)

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="quality", exclude_columns="wine_id")

    training_pd= training_set.load_df().toPandas()

    #Create train and test datasets

    X = training_pd.drop(["quality"], axis=1)
    Y = training_pd["quality"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Return the train and test datasets along with the training set
    return X_train, X_test, Y_train, Y_test, training_set

# COMMAND ----------

X_train, X_test, Y_train, Y_test, training_set = load_data(table_name, "wine_id")
X_train.head()

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
client = MlflowClient()

try: 
    client.delete_registered_model("wine_model") #delete the model if already exist
except:
    None

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Store

mlflow.sklearn.autolog(disable=True)

def train_model(X_train, X_test, Y_train, Y_test, training_set, fs):
    ## fit and log model
    rt = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
    rt.fit(X_train, Y_train)
    y_ped = rt.predict(X_test)

    mlflow.log_metric("test_mse", mean_squared_error(Y_test, y_ped))
    mlflow.log_metric("test_r2_score", r2_score(Y_test, y_ped))

    fs.log_model(
        model=rt,
        artifact_path="wine_quality_prediction",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name="wine_model")
    
train_model(X_train, X_test, Y_train, Y_test, training_set, fs)



# COMMAND ----------

batch_input_df = inference_data_df.drop("quality")  # Drop the lable column

predictions_df = fs.score_batch("models:/wine_model/latest", batch_input_df)  # Here we are using score_batch function of the Feature Store object 'fs' to make batch predictions on the model

display(predictions_df["wine_id", "prediction"])

# COMMAND ----------

so2_cols = ["free_sulfur_dioxide", "total_sulfur_dioxide"]

new_features_df= (features_df.withColumn("average_so2", expr("+".join(so2_cols))/2))

# COMMAND ----------

display(new_features_df)

# COMMAND ----------

fs.write_table(
    name=table_name,
    df=new_features_df,
    mode="merge"
)

# COMMAND ----------

display(fs.read_table(name=table_name))

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Store

mlflow.sklearn.autolog(disable=True)

def train_model(X_train, X_test, Y_train, Y_test, training_set, fs):
    ## fit and log model
    rt = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
    rt.fit(X_train, Y_train)
    y_ped = rt.predict(X_test)

    mlflow.log_metric("test_mse", mean_squared_error(Y_test, y_ped))
    mlflow.log_metric("test_r2_score", r2_score(Y_test, y_ped))

    fs.log_model(
        model=rt,
        artifact_path="wine_quality_prediction",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name="wine_model")
    
train_model(X_train, X_test, Y_train, Y_test, training_set, fs)

# COMMAND ----------

batch_input_df = inference_data_df.drop("quality")  # Drop the lable column

predictions_df = fs.score_batch("models:/wine_model/latest", batch_input_df)  # Here we are using score_batch function of the Feature Store object 'fs' to make batch predictions on the model

display(predictions_df["wine_id", "prediction"])
