# https://www.mlflow.org/docs/latest/python_api/mlflow.mleap.html


import mlflow
import mlflow.mleap
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

# training DataFrame
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0) ], ["id", "text", "label"])
# testing DataFrame
test_df = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")], ["id", "text"])

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
# Create an MLlib pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# log parameters
with mlflow.start_run():
  model = pipeline.fit(training)
  model.set_tag('sample execution')
  trest_metric = evaluator.evaluate(model.transform(test_df)
  mlflow.log_param("max_iter", 10)
  mlflow.log_param("reg_param", 0.001)
  mlflow.log_metric( 'test_'+ evaluator.getMetricName(),    trest_metric)               
  # log the Spark MLlib model in MLeap format
  mlflow.mleap.log_model(spark_model=model.bestModel, sample_input=test_df, artifact_path="mleap-model")
