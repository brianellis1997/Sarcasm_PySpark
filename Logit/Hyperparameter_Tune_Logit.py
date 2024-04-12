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


ss=SparkSession.builder.appName("Modeling Logistic Regression").getOrCreate()


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


# In[13]:


train = ss.read.csv("/storage/home/bje5256/work/Project/Train_Balanced.csv", header=True, schema=schema)
# In the cluster mode, we need to change to  `header=False` because it does not have header.


# In[ ]:





# # Data Preprocessing
# Our goals for data preprocessing are as follows:
# <br>**Feature Engineering:**<br><ul>
#     <li>Cyclic date time variables like `month`, `day_of_week`, `hour`</li>
#     <li>Counting text information like `word_count`, `punctuation_count`</li>
#     <li>Quantify sentiment of sarcastic vs non-sarcastic text.</li>
# </ul>
# <br>**Transformations:**<br>
# This will generally consist of transforming our categorical and text covariates into numeric features our model will be able to understand.<ul>
#     <li>One-hot-encoding `subreddit`</li>
#     <li>Possibly generating tf-idf vectors of `comment`, `parent_comment`, and `subreddit`</li>
# </ul>
# <br>**Scaling and Splitting:**<br><ul>
#     <li>Standardize our variables</li>
#     <li>Split our train dataset into train and validation 80/20</li>
# </ul>

# ## Feature Engineering
# Now that we have an idea of which variables are more important than the others, we can remove the unnecessary variables and add our feature engineered variables.

# In[14]:


# Import preprocessing libraries
from pyspark.sql import functions as F


# In[15]:


# Add date-time variables
df2 = train.withColumn('month', F.month('created_utc'))            .withColumn('day_of_week', F.dayofweek('created_utc'))            .withColumn('hour', F.hour('created_utc'))

# df2.show()


# In[16]:


# Calculate the number of nulls in each row by checking each column
null_check = df2.select([F.when(F.col(c).isNull(), 1).otherwise(0).alias(c) for c in df2.columns])

# Sum up the values across all columns for each row, resulting in a new DataFrame where each row has a sum of nulls
null_sums = null_check.withColumn('null_sum', sum(F.col(c) for c in null_check.columns))

# Filter to get only the rows with at least one null value and count them
num_rows_with_nulls = null_sums.filter(F.col('null_sum') > 0).count()

# print(f"Number of rows with at least one null value: {num_rows_with_nulls}")


# In[17]:


# num_rows_with_nulls/df2.count()


# In[18]:


# df2.count()


# Since the amount of rows with missing values is less than 1%, let's filter out these rows.

# In[19]:


df3 = df2.dropna()
# df3.count()


# In[20]:


from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import string

# text counting
count_punctuation_udf = udf(lambda comment: sum(1 for char in comment if char in string.punctuation) if comment is not None else 0, IntegerType())
count_capital_letters_udf = udf(lambda comment: sum(1 for char in comment if char.isupper()) if comment is not None else 0, IntegerType())

# Add columns for counting punctuation marks and capital letters
# comment
df3 = df3.withColumn('word_count', udf(lambda x: len(x.split()) if x is not None else 0, IntegerType())(col('comment')))
df3 = df3.withColumn('total_punctuation', count_punctuation_udf(col('comment')))

# df3.show()


# In[21]:


# Calculate the number of nulls in each row by checking each column
null_check = df3.select([F.when(F.col(c).isNull(), 1).otherwise(0).alias(c) for c in df3.columns])

# Sum up the values across all columns for each row, resulting in a new DataFrame where each row has a sum of nulls
null_sums = null_check.withColumn('null_sum', sum(F.col(c) for c in null_check.columns))

# Filter to get only the rows with at least one null value and count them
num_rows_with_nulls = null_sums.filter(F.col('null_sum') > 0).count()

# print(f"Number of rows with at least one null value: {num_rows_with_nulls}")


# In[22]:


# Drop unnecessary columns
# df3.columns


# In[23]:


clean_df = df3.select('label', 'comment', 'parent_comment', 'subreddit', 'score', 'month', 'day_of_week', 'hour', 'word_count', 'total_punctuation')
# clean_df.show()


# ## Transformations
# Now that we have all the necessary features in our `clean_df`, we can start processing the data and performing transformations such as one-hot-encoding and maybe creating tf-idf vectors to input into our traditional machine learning models.

# In[24]:


# Subreddit value counts
clean_df.select('subreddit').distinct().count()


# In[ ]:


# Step 1: Group by 'subreddit' and count the entries for each
subreddit_counts = clean_df.groupBy('subreddit').count()

# Step 2: Filter for subreddits with less than 5 comments
subreddits_less_than_5 = subreddit_counts.filter(col('count') < 2)

# Step 3: Count how many subreddits have less than 5 comments
number_of_subreddits_less_than_5 = subreddits_less_than_5.count()

# print(f"Number of subreddits with less than 5 comments: {number_of_subreddits_less_than_5}")


# In[ ]:


from pyspark.sql.functions import col, when
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline

# Assuming your DataFrame is named clean_df and you have a SparkSession initialized

# Step 1: Count comments per subreddit
subreddit_counts = clean_df.groupBy('subreddit').count()

# Step 2: Join this count back to the original dataframe to mark subreddits with less than 2 comments
clean_df = clean_df.join(subreddit_counts, on="subreddit", how="left")

# Replace subreddits with less than 2 comments with "other"
clean_df = clean_df.withColumn("subreddit_modified",
                               when(col("count") < 2, "other")
                               .otherwise(col("subreddit")))

# Drop the original 'subreddit' and 'count' columns as they are no longer needed
clean_df = clean_df.drop('subreddit', 'count')

# Rename 'subreddit_modified' back to 'subreddit' for clarity
clean_df = clean_df.withColumnRenamed("subreddit_modified", "subreddit")

# Step 3: One-hot-encode the modified subreddit column
# First, convert categories into indices
stringIndexer = StringIndexer(inputCol="subreddit", outputCol="subredditIndex")

# Then apply OneHotEncoder
encoder = OneHotEncoder(inputCols=["subredditIndex"], outputCols=["subredditVec"])

# Use a Pipeline to apply the steps
pipeline = Pipeline(stages=[stringIndexer, encoder])

# Fit and transform the data
model = pipeline.fit(clean_df)
encoded_df = model.transform(clean_df)

# Show the resulting DataFrame
# encoded_df.show()


# In[ ]:


# encoded_df.select('subredditVec').show()


# Vector Size (2718): This number represents the total number of unique subreddits (after processing, including the "other" category for subreddits with less than 2 comments) that have been identified across all comments in your dataset. It is the dimensionality of the one-hot encoded vector, meaning there are 2718 possible categories (subreddits) that each comment could belong to.
# 
# Index ([906], [3], [84], etc.): This number represents the index within the vector that corresponds to the specific subreddit a comment is associated with. The index starts at 0, so an index of 906 refers to the 907th subreddit in the sorted list of unique subreddits. Each comment's subreddit is represented by one of these indices, indicating which subreddit the comment belongs to.
# 
# Value ([1.0]): This indicates the value at the specified index. In the case of one-hot encoding, this will always be 1.0 for the index corresponding to the comment's subreddit, meaning the presence of that subreddit. All other positions in the vector will be 0 (not shown in the sparse vector representation), indicating the absence of those subreddits.

# In[ ]:


# We can drop the subreddit feature
transformed_df = encoded_df.drop('subreddit')
# transformed_df.show(5)


# In the context of the HashingTF transformer in PySpark, numFeatures specifies the number of features (or the size of the output feature vector) that you want to create for each document (in your case, each comment or parent_comment). This parameter is crucial for the "feature hashing" technique used by HashingTF.
# 
# Feature hashing, also known as the hashing trick, is a method to map potentially infinite-dimensional features (e.g., words in text data) to a finite-dimensional vector space using a hash function. The hash function converts words to indices in the feature vector, where each index corresponds to a "feature" or "bucket". The value at each index in the vector represents the frequency (term frequency, TF) of the words that hash to that index.
# 
# Pros: The primary advantage of feature hashing is its efficiency and scalability, as it allows for a fixed-size vector representation without needing to maintain a vocabulary in memory, which is particularly beneficial for large datasets.
# 
# Cons: A limitation of this approach is the possibility of hash collisions, where different words are mapped to the same index, especially if numFeatures is too small relative to the diversity of the corpus. While some collisions are generally acceptable and do not significantly impact model performance in practice, setting numFeatures too low can lead to a loss of information and potentially degrade model performance.
# 
# Using 
# 2^16
#   (or 65,536) as the number of features for a dataset with 800,000 rows can be a reasonable choice, especially when dealing with text data that can have a very large and sparse feature space. Here are a few considerations to keep in mind:
# 
# Dimensionality vs. Dataset Size
# Sufficient Dimensionality: For text data, which often involves a large vocabulary, having a sufficiently high dimensionality for the feature space is crucial to reduce the risk of hash collisions (where different words are mapped to the same feature index). A value of 
# 2
# 16
# 2 
# 16
#   offers a wide space that can accommodate a large vocabulary while keeping the collisions relatively low.
# Dataset Size: With 800,000 rows, your dataset is substantial. A larger numFeatures helps ensure that the nuanced differences in text across many samples can be captured without too much information loss due to collisions.
# Computational Considerations
# Memory and Speed: Larger numFeatures values will increase the memory usage and potentially the computation time for training models. However, Spark is designed to handle large-scale data processing, and feature vectors of size 
# 2
# 16
# 2 
# 16
#   are generally manageable on modern hardware, especially when using Spark's distributed computing capabilities.
# Model Performance: The choice of numFeatures can affect model performance. Too small a space might lead to too many collisions, losing important information and possibly degrading model performance. Conversely, an excessively large space might increase computational overhead without proportional gains in model accuracy. 
# 2
# 16
# 2 
# 16
#   is a good starting point, but it's always a good idea to experiment with different values if resources permit.

# In[ ]:


# Now let's create tf-idf vectors for our text comments
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline

# Tokenize comment and parent_comment
tokenizer_comment = Tokenizer(inputCol="comment", outputCol="comment_tokens")
tokenizer_parent_comment = Tokenizer(inputCol="parent_comment", outputCol="parent_comment_tokens")

# Apply HashingTF
hashingTF_comment = HashingTF(inputCol="comment_tokens", outputCol="rawFeatures_comment", numFeatures=2**13)
hashingTF_parent_comment = HashingTF(inputCol="parent_comment_tokens", outputCol="rawFeatures_parent_comment", numFeatures=2**13)

# Compute IDF for each feature vector
idf_comment = IDF(inputCol="rawFeatures_comment", outputCol="features_comment")
idf_parent_comment = IDF(inputCol="rawFeatures_parent_comment", outputCol="features_parent_comment")

# Build the pipeline
pipeline = Pipeline(stages=[tokenizer_comment, tokenizer_parent_comment, hashingTF_comment, hashingTF_parent_comment, idf_comment, idf_parent_comment])

# Fit the pipeline to the dataset
model = pipeline.fit(transformed_df)

# Transform the dataset
tfidf_df = model.transform(transformed_df)

# Show the transformed features
# tfidf_df.select("features_comment", "features_parent_comment").show(5)


# 65536: This is the size of the vector, determined by the numFeatures parameter you set in the HashingTF step. It represents the total number of distinct hash values that can be produced by the hashing function. Each possible hash value corresponds to a "bucket" that can hold the count of one or more words, depending on whether hash collisions occur.
# 
# [Indices]: These are the indices in the vector that have non-zero values. They represent the hash values of the words in the text, after the Tokenizer step has split the text into words and the HashingTF step has mapped these words to specific indices based on their hash values. Each index corresponds to a specific word (or multiple words in case of hash collisions).
# 
# [Values]: These are the TF-IDF scores for the words at the corresponding indices. The TF-IDF score is a measure of how important a word is to a document in a collection of documents. It increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the collection, which helps to adjust for the fact that some words appear more frequently in general.
# 
# Tokenization: Words, Not Characters
# The Tokenizer step in PySpark ML splits the text into words, not characters. So, the TF-IDF vectors represent the importance (weight) of each word within the text, not each character. The default behavior of the Tokenizer is to split the text by white spaces, effectively treating each contiguous string of characters separated by spaces as a word.
# Interpretation of the Vectors
# Each entry in these vectors corresponds to a word's weighted importance in the text, with the weight computed based on the term's frequency across the document and its inverse document frequency across all documents. This means:
# 
# Sparse Representation: Given that most documents contain only a small subset of the possible words, the TF-IDF vectors are sparse. This means that instead of storing a value for every possible word (which would be mostly zeros), it only stores values for words that actually appear in the text, significantly reducing memory usage.
# 
# Hashing and Collisions: Since HashingTF uses a fixed-size vector to represent an potentially unlimited vocabulary, multiple words can end up being hashed to the same index, leading to what's known as a hash collision. While this can introduce some noise into the data, the high dimensionality (e.g., 65536) helps to minimize the impact of these collisions on model performance.

# In[ ]:


# tfidf_df.show(5)


# In[ ]:


# tfidf_df.columns


# ## Feature Selection Insights
# 
# ### Potentially Useful Features
# 
# - **`label`**: Essential for supervised learning as it's the target variable we will predict.
# - **`score`, `month`, `day_of_week`, `hour`**: These features could provide useful signals for our model, depending on the nature of our task. For instance, the time of posting might correlate with certain types of comments or their reception.
# - **`word_count`, `total_punctuation`**: These could serve as proxies for the length or complexity of a comment, which might be relevant for some analyses.
# - **`features_comment`, `features_parent_comment`**: The TF-IDF vectors are likely to be highly informative for text analysis or natural language processing tasks, as they represent the textual content in a numerical form that models can work with.
# 
# ### Features to Review or Exclude
# 
# - **`subredditVec`**: This is the one-hot encoded representation of the subreddit. It's useful if we believe the subreddit context is important for our prediction task. However, we typically wouldn't need both `subredditVec` and `subredditIndex`.
# - **`subredditIndex`**: This is likely a numerical representation (index) of the subreddit used as an intermediate step for creating `subredditVec`. We would use either this or `subredditVec` for our model, not both, and `subredditVec` is usually the more useful form for machine learning models because it's one-hot encoded.
# 
# ### Intermediate Features (Usually Excluded from Modeling)
# 
# - **`comment_tokens`, `parent_comment_tokens`**: These are intermediate representations used in the process of generating TF-IDF vectors. They're the tokenized lists of words from the comments and are not usually used directly in modeling once we have the TF-IDF vectors.
# - **`rawFeatures_comment`, `rawFeatures_parent_comment`**: These represent the hashed feature vectors (before applying IDF) and are intermediate steps towards generating the `features_comment` and `features_parent_comment` TF-IDF vectors. We would typically use the final TF-IDF vectors for modeling, not these intermediate hash vectors.

# In[ ]:


# Now we can drop the comment and parent comment since they are represented as tf-idf vectors
final_df = tfidf_df.select('features_comment', 'features_parent_comment', 'subredditVec', 'score', 'month', 'day_of_week', 'hour', 'word_count', 'total_punctuation', 'label')
# final_df.columns


# In[ ]:


# final_df.show()


# Now, we can move onto scaling and splitting our data for modeling.

# # Scaling and Splitting

# In[ ]:


from pyspark.ml.feature import VectorAssembler

# List of numerical columns to scale
numericCols = ['score', 'month', 'day_of_week', 'hour', 'word_count', 'total_punctuation']

# Assemble numerical features into a vector
assembler = VectorAssembler(inputCols=numericCols, outputCol="numeric_features")
final_df = assembler.transform(final_df)


# In[ ]:


from pyspark.ml.feature import StandardScaler

# Scale the numerical features
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features", withStd=True, withMean=False)
scalerModel = scaler.fit(final_df)
final_df = scalerModel.transform(final_df)


# In[ ]:


# final_df.columns


# In[ ]:


scaled_df = final_df.select('features_comment', 
                            'features_parent_comment', 
                            'subredditVec', 
                            'scaled_numeric_features', 
                            'label')
# scaled_df.show(5)


# In[ ]:


train_df, val_df = scaled_df.randomSplit([0.8, 0.2], seed=22)

train_rows = train_df.count()
train_cols = len(train_df.columns)

val_rows = val_df.count()
val_cols = len(val_df.columns)

# print(f"Shape of train_df: ({train_rows}, {train_cols})")
# print(f"Shape of val_df: ({val_rows}, {val_cols})")


# ### Train and Evaluate using only text covars

# In[34]:


# from pyspark.ml.feature import VectorAssembler

# assembler = VectorAssembler(inputCols=["features_comment", "features_parent_comment"], outputCol="features")

# # Transform the dataset to include a new column 'features' that combines 'features_comment' and 'features_parent_comment'
# combined_df = assembler.transform(scaled_df)

# # Split the data into training and validation sets
# train_df, val_df = combined_df.randomSplit([0.8, 0.2], seed=22)


# # In[35]:


# from pyspark.ml.classification import LogisticRegression

# # Initialize the logistic regression model
# lr = LogisticRegression(featuresCol='features', labelCol='label')

# # Fit the model on the training data
# lrModel = lr.fit(train_df)


# # In[37]:


# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.mllib.evaluation import MulticlassMetrics
# from pyspark.sql.functions import col

# # Predict on the validation data
# predictions = lrModel.transform(val_df)

# # Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)
# print(f"Model Accuracy: {accuracy}")


# # In[38]:


# from pyspark.sql.types import FloatType

# # Convert predictions and labels to float type
# predictions = predictions.withColumn("label", predictions["label"].cast(FloatType()))
# predictions = predictions.withColumn("prediction", predictions["prediction"].cast(FloatType()))

# # Prepare the RDD required for MulticlassMetrics
# predictionAndLabels = predictions.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))

# # Instantiate metrics object
# metrics = MulticlassMetrics(predictionAndLabels)

# # Calculate precision, recall, and F1 Score
# precision = metrics.precision(1.0)
# recall = metrics.recall(1.0)
# f1Score = metrics.fMeasure(1.0, beta=1.0)

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1Score}")


# # # Text + Non-Text Covars

# # In[ ]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['features_comment', 
                            'features_parent_comment', 
                            'subredditVec', 
                            'scaled_numeric_features'], outputCol="features")

# Transform the dataset to include a new column 'features' that combines 'features_comment' and 'features_parent_comment'
combined_df = assembler.transform(scaled_df)

# Split the data into training and validation sets
train_df, val_df = combined_df.randomSplit([0.8, 0.2], seed=22)


# In[ ]:


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='label')

# Define a parameter grid to search over
paramGrid = (ParamGridBuilder() 
    .addGrid(lr.regParam, [0.01, 0.1, 1.0])  # List of C values to try
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])  # Elastic net parameter values
    .build())


# Define the evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# Define the cross-validator, which will automatically split the data, fit models, and evaluate them
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)  # 5-fold cross-validation

# Run the models through the cross-validator, which performs the parameter grid search
cvModel = cv.fit(train_df)

# Extract the best model from the CrossValidator
bestModel = cvModel.bestModel

# Predict on the validation data with the best model
predictions = bestModel.transform(val_df)

# Evaluate the best model's performance on the validation data
accuracy = evaluator.evaluate(predictions)

# Optionally, print the best regularization parameter
bestRegParam = bestModel._java_obj.getRegParam()
print(f"Best regularization parameter: {bestRegParam}")
print(f"Best model accuracy: {accuracy}")


