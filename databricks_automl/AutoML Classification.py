# Databricks notebook source
census_df = spark.sql("SELECT * FROM census_t")
display(census_df)

# COMMAND ----------

census_df.count()

# COMMAND ----------

train_df, test_df = census_df.randomSplit([0.95,0.05], seed=42)

# COMMAND ----------

display(train_df)
display(test_df)

# COMMAND ----------

from databricks import automl

# COMMAND ----------

automl.classify(train_df, target_col="income", timeout_minutes=5)
