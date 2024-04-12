#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pyspark


# ### Once we import pyspark, we need to use a `SparkContext`.  Every spark program needs a SparkContext object
# ### In order to use DataFrames, we also need to import `SparkSession` from `pyspark.sql`

# In[8]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType, TimestampType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row


# ## We then create a Spark Session variable (rather than Spark Context) in order to use DataFrame. 
# - Note: We temporarily use "local" as the parameter for master in this notebook so that we can test it in ICDS Roar Collab.  However, we need to remove "local" as usual to submit it to ICDS in cluster model (here make sure you remove ".master("local")" completely

# In[9]:

ss=SparkSession.builder.master("local").appName("Modeling Regression").getOrCreate()


# In[10]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[11]:


# # Clone repository
# !git clone https://brianellis1997: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@github.com/brianellis1997/Sarcasm_PySpark.git


# ## Load Data

# In[12]:


schema = StructType([
    StructField("ID", IntegerType(), False),
    StructField("label", IntegerType(), True),
    StructField("comment", StringType(), True),
    StructField("author", StringType(), True),
    StructField("subreddit", StringType(), True),
    StructField("score", IntegerType(), True),
    StructField("ups", IntegerType(), True),
    StructField("downs", IntegerType(), True),
    StructField("date", TimestampType(), True),
    StructField("created_utc", TimestampType(), True),
    StructField("parent_comment", StringType(), True)
])


# In[3]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Load the test data
train_df = ss.read.csv("/storage/home/bje5256/work/Project/Train_Balanced.csv", header=True, schema=schema)

train_df = train_df.sample(withReplacement=False, fraction=0.05, seed=42)

train = test_df.dropna()


from pyspark.ml.feature import RegexTokenizer, CountVectorizer, IDF, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import string

# Assume all initializations and schema definitions are done as per your setup

# Feature engineering functions
def get_month(dt):
    return dt.month
get_month_udf = udf(get_month, IntegerType())

def get_day_of_week(dt):
    return dt.dayofweek
get_day_of_week_udf = udf(get_day_of_week, IntegerType())

def get_hour(dt):
    return dt.hour
get_hour_udf = udf(get_hour, IntegerType())

def count_punctuations(text):
    return sum([1 for char in text if char in string.punctuation])
count_punctuations_udf = udf(count_punctuations, IntegerType())

# Define the stages of the pipeline
tokenizer_comment = RegexTokenizer(inputCol="comment", outputCol="comment_tokens", pattern="\\W")
tokenizer_parent_comment = RegexTokenizer(inputCol="parent_comment", outputCol="parent_comment_tokens", pattern="\\W")

hashingTF_comment = HashingTF(inputCol="comment_tokens", outputCol="rawFeatures_comment", numFeatures=2**13)
hashingTF_parent_comment = HashingTF(inputCol="parent_comment_tokens", outputCol="rawFeatures_parent_comment", numFeatures=2**13)

idf_comment = IDF(inputCol="rawFeatures_comment", outputCol="features_comment")
idf_parent_comment = IDF(inputCol="rawFeatures_parent_comment", outputCol="features_parent_comment")

# For subreddit, use StringIndexer + OneHotEncoder
stringIndexer_subreddit = StringIndexer(inputCol="subreddit", outputCol="subredditIndex")
encoder_subreddit = OneHotEncoder(inputCols=["subredditIndex"], outputCols=["subredditVec"])

# Combine all features into a single vector
assembler_features = VectorAssembler(inputCols=["features_comment", "features_parent_comment", "subredditVec"], outputCol="features")

# Define the full pipeline
preprocessingPipeline = Pipeline(stages=[
    tokenizer_comment, tokenizer_parent_comment,
    hashingTF_comment, hashingTF_parent_comment,
    idf_comment, idf_parent_comment,
    stringIndexer_subreddit, encoder_subreddit,
    assembler_features
])


# In[5]:


# Fit the pipeline on the training data
preprocessingModel = preprocessingPipeline.fit(train)

# Transform the training data
preprocessedTrain = preprocessingModel.transform(train)


# In[6]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Load the test data
test_df = ss.read.csv("/storage/home/bje5256/work/Project/Test_Balanced.csv", header=True, schema=schema)

test_df = test_df.sample(withReplacement=False, fraction=0.05, seed=42)

test = test_df.dropna()

# Transform the test data using the fitted preprocessing model
preprocessedTest = preprocessingModel.transform(test)


# In[7]:


# Initialize and fit the logistic regression model on the preprocessed training data
lr = LogisticRegression(featuresCol='features', labelCol='label', regParam=0.1)
lrModel = lr.fit(preprocessedTrain)


# In[8]:


# Predict on the preprocessed test data
test_predictions = lrModel.transform(preprocessedTest)

# Evaluate the model's accuracy on the test data
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
test_accuracy = evaluator.evaluate(test_predictions)
print("Accuracy on test data:", test_accuracy)


# In[ ]:




