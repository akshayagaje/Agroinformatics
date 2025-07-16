#spark packages
from pyspark import sql, SparkConf, SparkContext
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#initialize spark session
spark_session = sql.SparkSession.builder.appName("HDFS").getOrCreate()
spark_context = SparkContext.getOrCreate(SparkConf().setAppName("HDFS"))
logs = spark_context.setLogLevel("ERROR")
print("Spark session initialize")
#connection parameters for spark to Amazon S3
spark_session._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", "AKIAUGQ637RYKQTW47FH")
spark_session._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", "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")
spark_session._jsc.hadoopConfiguration().set("fs.s3a.impl","org.apache.hadoop.fs.s3native.NativeS3FileSystem")
spark_session._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
spark_session._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.BasicAWSCredentialsProvider")
spark_session._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.eu-west-1.amazonaws.com")
print("Connection to S3 Completed")
#code to load and display dataset from S3 using spark session object
dataset = spark_session.read.csv('s3n://cropyielddata/yield_df.csv', inferSchema=True, header=True)
dataset.show()
#describing dataset with details like count, mean, standard deviation of each dataset attributes
dataset.toPandas().describe()
#visualizing distribution of numerical data
dataset.toPandas().hist(figsize=(10, 8))
plt.title("Histogram distribution of dataset values")
plt.show()
#graph of different countries found in dataset for making crop yield
from pyspark.sql import functions
areas = dataset.select('Area').filter(functions.col('Area').isNotNull()).toPandas().values.ravel()
names, count = np.unique(areas, return_counts = True)
height = count
bars = names
y_pos = np.arange(len(bars))
plt.figure(figsize = (14, 3)) 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Different Area Graph for Crop Yield")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()
#visualizing graph of different crops found in dataset
from pyspark.sql import SQLContext
sql = SQLContext(spark_session)
#converting spark dataframe to spark sql query
dataset.registerTempTable("crop")
df = sql.sql("SELECT Item from crop")
df = df.toPandas()
unique, count = np.unique(df['Item'], return_counts=True)
values = []
for i in range(len(unique)):
    values.append([unique[i], count[i]])
values = pd.DataFrame(values, columns = ['Crop', 'Count'])   
plt.figure(figsize=(8,3))
sns.barplot(x='Crop',y='Count', data=values)
plt.title('Most Common Crop Yield by Different Countries')
plt.xticks(rotation=90)
plt.show()
#query to visualize different yield of crop by different countries
df = sql.sql("SELECT Area, Item, yield from crop")
df = df.toPandas()
data = df.groupby(['Item', 'Area'])['yield'].sum().sort_values(ascending=False).nlargest(30).reset_index()
sns.catplot(x="Item", y="yield", hue='Area', data=data, kind='point')
plt.title("Crop Yield Graphs of Different Countries")
plt.xticks(rotation=90)
plt.show()
#graph of Top 20 highest average rainfall area wise 
df = sql.sql("SELECT Area, average_rain_fall_mm_per_year from crop")
df = df.toPandas()
df = df.groupby('Area')['average_rain_fall_mm_per_year'].mean().sort_values(ascending=False).nlargest(20).reset_index()
plt.figure(figsize=(8,4))
plt.plot(df['Area'], df['average_rain_fall_mm_per_year'])
plt.title("Top 20 Area Wise Average Rainfall Graph")
plt.xticks(rotation=90)
plt.show()
#graph of Top 20 highest area wise pesticides consumption 
df = sql.sql("SELECT Area, pesticides_tonnes from crop")
df = df.toPandas()
df = df.groupby('Area')['pesticides_tonnes'].mean().sort_values(ascending=False).nlargest(20).reset_index()
plt.figure(figsize=(8,4))
plt.plot(df['Area'], df['pesticides_tonnes'])
plt.title("Top 20 Area Wise Average Pesticides Consumption Graph")
plt.xticks(rotation=90)
plt.show()
#split dataset into train and test where application using 80% dataset for training and 20% for testing
#extracting train and test features from dataset
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer
#extracing data for selected country and crop and in above line we are getting all maize yield from India
df = sql.sql("SELECT * from crop where Area='India' and Item='Maize'")#==============
#converting Area and Item column from string to numeric vector
indexer = StringIndexer(inputCol="Area", outputCol="AreaEncode")
encoder = indexer.fit(df)
df = encoder.transform(df)
#converting Area and Item column from string to numeric vector
indexer = StringIndexer(inputCol="Item", outputCol="ItemEncode")
encoder = indexer.fit(df)
df = encoder.transform(df)
#giving required features for training to select
requiredColumns = ['Year', 'yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes','avg_temp', 'AreaEncode', 'ItemEncode']
vec_assembler = VectorAssembler(inputCols=requiredColumns, outputCol='train',handleInvalid="skip")
transformed = vec_assembler.transform(df)
indexer = StringIndexer(inputCol="yield",outputCol="predict",handleInvalid="skip")
transformed = indexer.fit(transformed).transform(transformed)
#normalizing extracted crop features
scaler = MinMaxScaler(inputCol="train", outputCol="scaled_train")
transformed = scaler.fit(transformed).transform(transformed)
#splitting dataset into train and test
(X_train, X_test) = transformed.randomSplit([0.8, 0.2])
print("80% dataset for training : "+str(X_train.count()))
print("20% dataset for testing  : "+str(X_test.count()))
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
#creating object of decision tree algorithm
dt = DecisionTreeRegressor(featuresCol="scaled_train", labelCol = 'predict')
#training DT on selected train data
dt_model = dt.fit(X_train)
#perform prediction on test data
predict = dt_model.transform(X_test)
#collect original crop yield
true = predict.select(['predict']).collect()
#collect predicted crop yield
pred = predict.select(['prediction']).collect()
#calculate decision tree performance using RMSE meric 
evaluator = RegressionEvaluator(labelCol="predict", predictionCol="prediction", metricName="rmse")
rmse_error = evaluator.evaluate(predict)
print("Decision Tree RMSE = "+str(rmse_error)+"\n")
#plot graph of true and predicted crop yield
trueYield = []
predictedYield = []
for i in range(0, 100): 
    trueYield.append(true[i].predict*100)
for i in range(0, 100): 
    predictedYield.append(pred[i].prediction*100)
for i in range(0, 20):
    print("True Yield = "+str(trueYield[i])+" Decision Tree Predicted Yield = "+str(predictedYield[i]))
plt.plot(trueYield, color = 'red', label = 'Original Crop Yield')
plt.plot(predictedYield, color = 'green', label = 'Decision Tree Crop Yield')
plt.title('Decision Tree True & Predicted Crop Yield Graph')
plt.xlabel('Test Data')
plt.ylabel('Crop Yield')
plt.legend()
plt.show()        
from pyspark.ml.regression import LinearRegression
#training linear regression on train features of crop yield dataset
lr = LinearRegression(featuresCol="scaled_train", labelCol = 'predict')
lr_model = lr.fit(X_train)
predict = lr_model.transform(X_test)
#collect original crop yield
true = predict.select(['predict']).collect()
#collect predicted crop yield
pred = predict.select(['prediction']).collect()
#calculate linear regression performance using RMSE meric 
evaluator = RegressionEvaluator(labelCol="predict", predictionCol="prediction", metricName="rmse")
lr_rmse_error = evaluator.evaluate(predict)
print("Linear Regression RMSE = "+str(lr_rmse_error)+"\n")
#plot graph of true and predicted crop yield
trueYield = []
predictedYield = []
for i in range(0, 100): 
    trueYield.append(true[i].predict*100)
for i in range(0, 100): 
    predictedYield.append(pred[i].prediction*100)
for i in range(0, 20):
    print("True Yield = "+str(trueYield[i])+" Linear Regression Predicted Yield = "+str(predictedYield[i]))
plt.plot(trueYield, color = 'red', label = 'Original Crop Yield')
plt.plot(predictedYield, color = 'green', label = 'Linear Regression Crop Yield')
plt.title('Linear Regression True & Predicted Crop Yield Graph')
plt.xlabel('Test Data')
plt.ylabel('Crop Yield')
plt.legend()
plt.show()
#RMSE comaprison Graph
height = [rmse_error, lr_rmse_error]
bars = ['Decision Tree TMSE', 'Linear Regression RMSE']
y_pos = np.arange(len(bars))
plt.figure(figsize = (4, 3)) 
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.xlabel("Algorithm Names")
plt.ylabel("RMSE")
plt.title("RMSE Comparison Graph")
plt.show()
