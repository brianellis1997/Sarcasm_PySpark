#!/usr/bin/env python
# coding: utf-8

# # DS/CMPSC 410 Spring 2024
# # Instructor: Professor Romit Maulik
# # TAs: Peng Jin, Haiwen Guan
# 
# # Project Test: Loading in Data and storing in DataFrame
# 
# ### Brian Ellis

# ## The first thing we need to do in each Jupyter Notebook running pyspark is to import pyspark first.

# In[2]:


import pyspark


# ### Once we import pyspark, we need to use a `SparkContext`.  Every spark program needs a SparkContext object
# ### In order to use DataFrames, we also need to import `SparkSession` from `pyspark.sql`

# In[3]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType, TimestampType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row


# ## We then create a Spark Session variable (rather than Spark Context) in order to use DataFrame. 
# - Note: We temporarily use "local" as the parameter for master in this notebook so that we can test it in ICDS Roar Collab.  However, we need to remove "local" as usual to submit it to ICDS in cluster model (here make sure you remove ".master("local")" completely

# In[4]:


ss=SparkSession.builder.appName("Project Test").getOrCreate()


# In[5]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[6]:


# # Clone repository
#git clone https://brianellis1997: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@github.com/brianellis1997/Sarcasm_PySpark.git


# ## Load Data

# In[7]:


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


# In[8]:


train = ss.read.csv("/storage/home/alw6312/work/Project/Train_Balanced.csv", header=False, schema=schema)
# In the cluster mode, we need to change to  `header=False` because it does not have header.


# In[9]:


train.show(5)


# In[10]:


train.count()


# In[11]:


train.printSchema()


# ## Subsample for editing in Local Mode

# In[12]:


# Sample 10% of the DataFrame
df = train.sample(withReplacement=False, fraction =0.99, seed=42)


# In[13]:





# ## Text only

# In[15]:


comment = df.select("comment")



# In[16]:


parent_comment = df.select('parent_comment')



# In[17]:


rdd_1 = comment.rdd
rdd_2 = parent_comment.rdd


# In[18]:


comment_flat = rdd_1.flatMap(lambda x: x['comment'].lower().split(' '))



# In[19]:


# Remove small words
min_length = 8
tokens_rdd = comment_flat.filter(lambda line: len(line) > min_length)


# In[20]:


# Count each word
num_partitions = 5
kvRDD = tokens_rdd.map(lambda x: (x, 1))
kvRDD_reduced = kvRDD.reduceByKey(lambda x, y: x+y)


# In[21]:





# In[22]:


# Sort by most common words
kvRDD_sorted = kvRDD_reduced.sortBy(lambda x: x[1], ascending=False)


# In[23]:


kvRDD_sorted.take(10)


# ## Sarcastic vs Non-Sarcastic counts

# Now that we know how to filter words based on length and sort by words based on prevelance, let's compare the counts of unique words in sarcastic and non-sarcastic comments respectivelY and see if there's a difference.

# In[24]:


sarcastic = df[df['label']==1]
sarcastic.count()


# In[25]:


non_sarcastic = df[df['label']==0]
non_sarcastic.count()


# In[26]:


sarc_comment = sarcastic.select("comment")



# In[27]:


non_sarc_comment = non_sarcastic.select('comment')



# In[28]:


rdd_1 = sarc_comment.rdd
rdd_2 = non_sarc_comment.rdd


# In[29]:


sarc_flat = rdd_1.flatMap(lambda x: x['comment'].lower().split(' '))
non_sarc_flat = rdd_2.flatMap(lambda x: x['comment'].lower().split(' '))


# In[30]:





# In[32]:


# Remove small words
min_length = 8
sarc_tokens_rdd = sarc_flat.filter(lambda line: len(line) > min_length)
non_sarc_tokens_rdd = non_sarc_flat.filter(lambda line: len(line) > min_length)


# In[33]:


# Count each word for sarcastic and non-sarcastic respectively
num_partitions = 5
sarc_kvRDD = sarc_tokens_rdd.map(lambda x: (x, 1))
sarc_kvRDD_reduced = sarc_kvRDD.reduceByKey(lambda x, y: x+y)

non_sarc_kvRDD = non_sarc_tokens_rdd.map(lambda x: (x, 1))
non_sarc_kvRDD_reduced = non_sarc_kvRDD.reduceByKey(lambda x, y: x+y)


# In[34]:


sarc_kvRDD_reduced.take(3)


# In[35]:


non_sarc_kvRDD_reduced.take(3)


# In[36]:


# Sort by most common words
#sarc_kvRDD_sorted = sarc_kvRDD_reduced.sortBy(lambda x: x[1], ascending=False)
#non_sarc_kvRDD_sorted = non_sarc_kvRDD_reduced.sortBy(lambda x: x[1], ascending=False)


# In[37]:


# Sarcastic words
#sarc_kvRDD_sorted.take(10)


# In[38]:


# Non-Sarcastic words
#non_sarc_kvRDD_sorted.take(10)


# <br>We see some differences between the most common sarcastic vs. non-sarcastic words. The most frequent non-sarcastic word is a reference to a person 'Gjallarhorn' and the most frequent sarcastic word is 'obviously'.<br><br>We can do several things now for analysis, such as looking at the difference in word frequency in parent comments between sarcastic and non-sarcastic comments, we can visualize the distributions of the other variables with respect to their sarcastic label, and we can also quantify the sentiment of sarcastic vs non-sarcastic comments (maybe sarcastic comments haver greater negative sentiment?). Along the lines of this sentiment analysis, we can feature engineer this sentiment as a novel variable and maybe look into engineering other variables e.g. punctuation count.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # EDA
# In this section, we will systematically go through each of our non-text variables to understand their relationships between eachother, our label, and their statistical significance. This will allow us to understand which variables to include or not in our modelling.

# ## Non-Text Covariates
# First we will analyze and investigate our non-text covariates and their relationship and possible utility to our outcome variable `label`.

# ### Subreddit Analysis

# Which subreddits contain the most sarcastic comments?

# In[39]:


df.printSchema()


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# Calculate the count of sarcastic comments per subreddit
subreddit_count = sarcastic.groupBy('subreddit').count()

# Collect the top 20 subreddits by count
top = subreddit_count.orderBy(F.desc('count')).limit(20).toPandas()

# Plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=top, x='count', y='subreddit')
plt.xlabel('Number of Sarcastic Comments')
plt.ylabel('Subreddit')
plt.title('Frequency of Sarcastic Comments by Subreddit')
plt.show()


# In[41]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# Calculate the count of sarcastic comments per subreddit
sarcastic_count = sarcastic.groupBy('subreddit').count()
non_sarcastic_count = non_sarcastic.groupBy('subreddit').count()

# Collect the top 20 subreddits by count for both sarcastic and non-sarcastic comments
top_sarcastic = sarcastic_count.orderBy(F.desc('count')).limit(20).toPandas()
top_non_sarcastic = non_sarcastic_count.orderBy(F.desc('count')).limit(20).toPandas()

# Merge sarcastic and non-sarcastic counts based on subreddit
merged_counts = top_sarcastic.merge(top_non_sarcastic, on='subreddit', suffixes=('_sarcastic', '_non_sarcastic'))

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
r1 = np.arange(len(merged_counts))
r2 = [x + bar_width for x in r1]

# Plot side-by-side bar charts
plt.figure(figsize=(12, 6))

# Plot sarcastic comments
plt.bar(r1, merged_counts['count_sarcastic'], color='skyblue', width=bar_width, edgecolor='grey', label='Sarcastic')

# Plot non-sarcastic comments
plt.bar(r2, merged_counts['count_non_sarcastic'], color='salmon', width=bar_width, edgecolor='grey', label='Non-Sarcastic')

# Add xticks and labels
plt.xlabel('Subreddit', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(merged_counts))], merged_counts['subreddit'], rotation=90)

# Add ylabel and title
plt.ylabel('Number of Comments', fontweight='bold')
plt.title('Frequency of Sarcastic and Non-Sarcastic Comments by Subreddit', fontweight='bold')

# Add legend
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# We see that there are some differences in sarcastic comment prevelance between subreddits. In particular, political subreddits such as `politics` and `worldnews` have a greater proportion of sarcastic vs non-sarcastic comments. This political sentiment might be something we can feature engineer out of our dataset.

# In[42]:


import seaborn as sns
import re

# Define regex patterns for news and politics
news_politics_regexes = [
    'news',
    'politics',
    'worldnews',
    'worldpolitics',
    'uspolitics',
    'political',
    'currentevents',
    'government',
    'internationalnews',
    'nationalpolitics',
    'usnews',
    'worldaffairs',
    'politicaldiscussion',
    'uspolitics',
    'politicsdebate',
    # Add more patterns as needed
]

# Compile regex patterns
news_politics_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in news_politics_regexes]


# In[43]:


# Convert compiled regex patterns to strings
news_politics_patterns_str = '|'.join(pattern.pattern for pattern in news_politics_patterns)

# Filter DataFrame for subreddits containing keywords using regex matching
news_politics_comments = df.filter(df['subreddit'].rlike(news_politics_patterns_str))

# Plot the count of sarcastic and non-sarcastic comments for subreddits related to news or politics
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=news_politics_comments.toPandas(), palette='Set1')
plt.title('Distribution of Sarcastic and Non-sarcastic Comments for News/Politics Subreddits')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Non-Sarcastic', 'Sarcastic'])
plt.show()


# In[44]:


# Group by subreddit and count the occurrences
subreddit_counts = news_politics_comments.groupBy('subreddit').count()

# Get the top 10 subreddits with the most comments
top_subreddits = subreddit_counts.orderBy(F.desc('count')).limit(10).select('subreddit').rdd.flatMap(lambda x: x).collect()

# Filter the DataFrame to include only the top 10 subreddits
top_subreddit_comments = news_politics_comments.filter(news_politics_comments['subreddit'].isin(top_subreddits))

# Group by subreddit and label, and count the occurrences
counts = top_subreddit_comments.groupBy('subreddit', 'label').count().orderBy('subreddit')

# Convert the result to Pandas DataFrame for plotting
counts_df = counts.toPandas()

# Pivot the DataFrame to have labels as columns
pivot_df = counts_df.pivot(index='subreddit', columns='label', values='count')

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Plot sarcastic and non-sarcastic comments side by side
pivot_df.plot(kind='bar', stacked=False, ax=ax)

plt.title('Distribution of Sarcastic and Non-sarcastic Comments for Top 10 News/Politics Subreddits')
plt.xlabel('Subreddit')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Label', labels=['Non-Sarcastic', 'Sarcastic'])
plt.tight_layout()
plt.show()


# In[ ]:





# ### Author

# In[45]:


df.columns


# In[46]:


author = df.select('author', 'label')
author.show()


# In[47]:


from pyspark.sql import functions as F

# Group by 'author' and 'label', and count occurrences
author_counts = author.groupBy('author', 'label').count()

# Pivot the DataFrame to have labels as columns
author_pivot = author_counts.groupBy('author').pivot('label').agg(F.first('count')).fillna(0)

# Show the pivoted DataFrame
author_pivot.show()


# In[48]:


# Group by 'author' and 'label', and count occurrences
author_counts = author.groupBy('author', 'label').count()

# Pivot the DataFrame to have labels as columns
author_pivot = author_counts.groupBy('author').pivot('label').agg(F.first('count')).fillna(0)

# Calculate the total count for each author
author_pivot = author_pivot.withColumn('total_count', sum(author_pivot[col] for col in ['0', '1']))

# Sort the DataFrame by the total count in descending order
author_pivot_sorted = author_pivot.orderBy(F.desc('total_count'))

# Show the top 10 most sarcastic authors
top_sarcastic_authors = author_pivot_sorted.filter(author_pivot_sorted['1'] > 0).head(10)

# Show the top 10 most non-sarcastic authors
top_non_sarcastic_authors = author_pivot_sorted.filter(author_pivot_sorted['0'] > 0).head(10)


# In[49]:


# Show the top 10 most sarcastic authors
print("Top 10 most sarcastic authors by count:")
for row in top_sarcastic_authors:
    print(f"{row['author']}: {row['1']}")

# Show the top 10 most non-sarcastic authors
print("\nTop 10 most non-sarcastic authors by count:")
for row in top_non_sarcastic_authors:
    print(f"{row['author']}: {row['0']}")


# If we look closely, we see that the same authors who posted the greatest number of sarcastic comments, also posted the greatest number of non-sarcastic comments. This tells us the `author` variable isn't important for determining our label.

# ### Numeric Variables (Ups, Downs, Score)

# In[50]:


numeric_df = df.select(['ups', 'downs', 'score', 'label'])
numeric_df.show(5)


# In[51]:


numeric_df.count()


# In[52]:


from pyspark.sql.functions import col

# Filter out rows containing null values in 'ups', 'downs', and 'score' columns
numeric_df = numeric_df.filter(col('ups').isNotNull() & col('downs').isNotNull() & col('score').isNotNull())

numeric_df.count()


# In[53]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import corr, round

# Compute the correlation between each numeric variable and the label
correlation_with_label = numeric_df.select(
    round(corr("ups", "label"), 3).alias("ups_label_corr"),
    round(corr("downs", "label"), 3).alias("downs_label_corr"),
    round(corr("score", "label"), 3).alias("score_label_corr")
)

# Show the correlation coefficients
correlation_with_label.show()


# We see not much correlation between the numeric variables and the sarcastic label.

# In[54]:


import matplotlib.pyplot as plt

# Create separate DataFrames for each label value
label_0_df = numeric_df.filter(df['label'] == 0)
label_1_df = numeric_df.filter(df['label'] == 1)

# Define histogram bins
bins = 20

# Create separate plots for ups, downs, and score
plt.figure(figsize=(15, 5))

# Plot histogram for ups
plt.subplot(1, 3, 1)
plt.hist(label_0_df.select('ups').rdd.flatMap(lambda x: x).collect(), bins=bins, alpha=0.2, color='blue', label='Label 0')
plt.hist(label_1_df.select('ups').rdd.flatMap(lambda x: x).collect(), bins=bins, alpha=0.2, color='red', label='Label 1')
plt.xlabel('Ups')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(xmin=0, xmax=400)

# Plot histogram for downs
plt.subplot(1, 3, 2)
plt.hist(label_0_df.select('downs').rdd.flatMap(lambda x: x).collect(), bins=bins, alpha=0.2, color='blue', label='Label 0')
plt.hist(label_1_df.select('downs').rdd.flatMap(lambda x: x).collect(), bins=bins, alpha=0.2, color='red', label='Label 1')
plt.xlabel('Downs')
plt.ylabel('Frequency')
plt.legend()

# Plot histogram for score
plt.subplot(1, 3, 3)
plt.hist(label_0_df.select('score').rdd.flatMap(lambda x: x).collect(), bins=bins, alpha=0.2, color='blue', label='Label 0')
plt.hist(label_1_df.select('score').rdd.flatMap(lambda x: x).collect(), bins=bins, alpha=0.2, color='red', label='Label 1')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(xmin=0, xmax=800)

plt.tight_layout()
plt.show()


# We can see that non-sarcastic comments (label 0) receive a greater amount of upvotes, however they also receive a higher amount oof downvotes. This results in a score variable that has non-sarcastic comments in the upper and lower percentiles but sarcastic comments in the middle.

# In[55]:


# Plot histogram for 'score' column
plt.figure(figsize=(8, 6))
plt.hist(numeric_df.filter(numeric_df['label'] == 0).select('score').rdd.flatMap(lambda x: x).collect(), bins=20, color='blue', alpha=0.25, label='Non-sarcastic')
plt.hist(numeric_df.filter(numeric_df['label'] == 1).select('score').rdd.flatMap(lambda x: x).collect(), bins=20, color='red', alpha=0.25, label='Sarcastic')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of Score by Label')
plt.legend()
plt.grid(True)
plt.xlim(xmin=-150, xmax=300)  # Set the x-axis limits
plt.show()


# Because it seems `score` is simply a linear combination of `ups` and `downs`, it would be safe to say we can remove these dependent variables and just keep `score` while still retaining the necessary information.

# ### Date Time Variables

# In[56]:


time = df.select(['date', 'created_utc', 'label'])
time.show(5)


# Since the year, month, and day are already included in `created_utc`, the `date` variable is redundant and can be dropped.

# In[57]:


time = time.select(['created_utc', 'label'])
time.show(5)


# In[58]:


# Extract year, month, day of the week, and hour from created_utc
time = time.withColumn('year', F.year('created_utc'))            .withColumn('month', F.month('created_utc'))            .withColumn('day_of_week', F.dayofweek('created_utc'))            .withColumn('hour', F.hour('created_utc'))


# In[59]:


time.show(5)


# In[60]:


time.count()


# In[61]:


# Filter missing values
time = time[time['created_utc'].isNotNull()]
time.count()


# In[62]:


# Group by year and label, and count occurrences
year_distribution = time.groupBy('year', 'label').count().orderBy('year')

# Convert the result to Pandas DataFrame for plotting
year_df = year_distribution.toPandas()

# Pivot the DataFrame to have labels as columns
year_pivot_df = year_df.pivot(index='year', columns='label', values='count')

# Plotting
year_pivot_df.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title('Distribution of Sarcastic and Non-sarcastic Comments by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Label', labels=['Non-Sarcastic', 'Sarcastic'])
plt.tight_layout()
plt.show()


# In[63]:


# Group by month and label, and count occurrences
month_distribution = time.groupBy('month', 'label').count().orderBy('month')

# Convert the result to Pandas DataFrame for plotting
month_df = month_distribution.toPandas()

# Pivot the DataFrame to have labels as columns
month_pivot_df = month_df.pivot(index='month', columns='label', values='count')

# Plotting
month_pivot_df.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title('Distribution of Sarcastic and Non-sarcastic Comments by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Label', labels=['Non-Sarcastic', 'Sarcastic'])
plt.tight_layout()
plt.show()


# In[64]:


# Group by day of the week and label, and count occurrences
month_distribution = time.groupBy('day_of_week', 'label').count().orderBy('day_of_week')

# Convert the result to Pandas DataFrame for plotting
month_df = month_distribution.toPandas()

# Pivot the DataFrame to have labels as columns
month_pivot_df = month_df.pivot(index='day_of_week', columns='label', values='count')

# Plotting
month_pivot_df.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title('Distribution of Sarcastic and Non-sarcastic Comments by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Label', labels=['Non-Sarcastic', 'Sarcastic'])
plt.tight_layout()
plt.show()


# In[65]:


# Group by hour and label, and count occurrences
month_distribution = time.groupBy('hour', 'label').count().orderBy('hour')

# Convert the result to Pandas DataFrame for plotting
month_df = month_distribution.toPandas()

# Pivot the DataFrame to have labels as columns
month_pivot_df = month_df.pivot(index='hour', columns='label', values='count')

# Plotting
month_pivot_df.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title('Distribution of Sarcastic and Non-sarcastic Comments by Hour')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Label', labels=['Non-Sarcastic', 'Sarcastic'])
plt.tight_layout()
plt.show()


# ## Text Covariates
# Here we dive into looking at our text covars `comment` and `parent_comment`.

# In[66]:


text_df = df.select(['parent_comment', 'comment', 'label'])
text_df.show(5, truncate=False)


# In[67]:


from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import string

# Define UDFs for counting punctuation marks and capital letters
count_punctuation_udf = udf(lambda comment: sum(1 for char in comment if char in string.punctuation) if comment is not None else 0, IntegerType())
count_capital_letters_udf = udf(lambda comment: sum(1 for char in comment if char.isupper()) if comment is not None else 0, IntegerType())

# Add columns for counting punctuation marks and capital letters
# comment
text_df = text_df.withColumn('capital_letters_count', count_capital_letters_udf(col('comment')))
text_df = text_df.withColumn('word_count', udf(lambda x: len(x.split()) if x is not None else 0, IntegerType())(col('comment')))
text_df = text_df.withColumn('total_punctuation', count_punctuation_udf(col('comment')))

# Parent comment
text_df = text_df.withColumn('parent_capital_letters_count', count_capital_letters_udf(col('parent_comment')))
text_df = text_df.withColumn('parent_word_count', udf(lambda x: len(x.split()) if x is not None else 0, IntegerType())(col('parent_comment')))
text_df = text_df.withColumn('parent_total_punctuation', count_punctuation_udf(col('parent_comment')))

# Convert PySpark DataFrame to pandas DataFrame
df_pandas = text_df.select('capital_letters_count', 'word_count', 
                      'total_punctuation', 
                      'parent_capital_letters_count', 'parent_word_count', 'parent_total_punctuation', 'label').toPandas()



# Adjust the columns_to_plot to match your DataFrame's structure
columns_to_plot = ['capital_letters_count', 'word_count', 'total_punctuation',
                   'parent_capital_letters_count', 'parent_word_count', 'parent_total_punctuation']

# Filter out outliers for each column
for column in columns_to_plot:
    lower_bound, upper_bound = np.percentile(df_pandas[column], [2.5, 97.5])
    df_pandas = df_pandas[(df_pandas[column] >= lower_bound) & (df_pandas[column] <= upper_bound)]


# Filter out outliers for each column
for column in columns_to_plot:
    lower_bound, upper_bound = np.percentile(df_pandas[column], [2.5, 97.5])
    # Update df_pandas to exclude outliers for each column
    df_pandas = df_pandas[(df_pandas[column] >= lower_bound) & (df_pandas[column] <= upper_bound)]


# In[68]:


df_pandas.head(20)


# In[69]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Determine the number of rows needed for the subplot. Calculate based on the number of non-parent columns
num_features = len([col for col in df_pandas.columns if not col.startswith('parent_')])
num_rows = (num_features + 2) // 3  # Adjust the denominator to change the number of columns per row

plt.figure(figsize=(20, 6 * num_rows))  # Adjust figure size based on the number of rows

# Loop through each feature, ignoring 'parent_' columns directly
for i, col in enumerate([col for col in df_pandas.columns if not col.startswith('parent_') and col != 'label']):
    plt.subplot(num_rows, 3, i + 1)
    # Melting the DataFrame to long format for the current pair of columns (comment and parent_comment)
    melted_df = pd.melt(df_pandas[[col, 'parent_' + col]], var_name='variable', value_name='value')
    sns.boxplot(x='variable', y='value', data=melted_df)
    plt.title(col)
    plt.xlabel('Variable')
    plt.xticks(rotation=45)  # Adjust rotation if necessary
    plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting histograms for each feature
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_pandas, x=column, hue="label", element="step", stat="density", common_norm=False)
    plt.title(f'Distribution of {column} by Label')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(title='Label', labels=['Non-sarcastic', 'Sarcastic'])
    plt.show()

# Box plots
plt.figure(figsize=(12, 8))
df_melted = df_pandas.melt(id_vars=['label'], value_vars=columns_to_plot, var_name='Features', value_name='Count')
sns.boxplot(data=df_melted, x='Features', y='Count', hue='label')
plt.xticks(rotation=45)
plt.title('Box Plot of Features by Label')
plt.show()


# Judging from these plots, it seems like `total_punctuation` has the greatest variance between labels. Word count for both child and parent comments also looks like there's a discrepency but it's not convincing.<br>Let's determine the statistical significance of each feaure with `label`.

# In[71]:


df_pandas.isnull().sum()


# In[72]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Splitting the DataFrame into X (features) and y (target variable)
X = df_pandas.drop('label', axis=1)  # Features
y = df_pandas['label']  # Target variable

# Splitting the data into training and test sets for model validation (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data, then convert it back to a DataFrame
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

# Transform the test data, then convert it back to a DataFrame
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)


# In[73]:


#from statsmodels.regression.linear_model import OLS

# Adding a constant to the model (intercept)
#X_train_scaled_sm = sm.add_constant(X_train_scaled)

# Fit the OLS model
#model = OLS(y_train, X_train_scaled_sm).fit()

# View the summary for p-values and other statistics
#print(model.summary())


# Looking at the t-statistics and p-values, we can see that `word_count`, `total_punctuation`, and `parent_word_count` are statistically significant. We should include these variables in our model. However, `parent_word_count` and `word_count` might be redundant and the comments word count contains a highly significant t-statistic, therefore we will only keep the child's word count.

# ## LDA
# We will try LDA for topic modeling

# In[74]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.clustering import LDA


# In[75]:


lda_df = df.select('ID', 'comment')
lda_df.show()


# In[76]:


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the stopwords
stop_words = set(stopwords.words('english'))

# Define a function to remove both stopwords and punctuation
def remove_stopwords_and_punctuation(text):
    if text is not None:
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize the text
        words = word_tokenize(text)
        # Filter stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words]
        # Rejoin the filtered words back into a string
        return " ".join(filtered_words)
    else:
        return None

# Register the function as a UDF
remove_stopwords_and_punctuation_udf = udf(remove_stopwords_and_punctuation, StringType())

# Apply the UDF to remove stopwords and punctuation from the 'comment' column
lda_df = lda_df.withColumn('comment_clean', remove_stopwords_and_punctuation_udf(lda_df['comment']))

# Show the result
lda_df.select('comment', 'comment_clean').show(truncate=False)


# In[77]:


# Tokenization
tokenizer = Tokenizer(inputCol='comment_clean', outputCol='words')
data_words = tokenizer.transform(lda_df)


# In[78]:


cv = CountVectorizer(inputCol='words', outputCol='features', vocabSize=1000, minDF=50.0)
cv_model = cv.fit(data_words)
data_features = cv_model.transform(data_words)


# In[79]:


# LDA model training
num_topics = 10
lda = LDA(k=num_topics, maxIter=10)
lda_model = lda.fit(data_features)


# In[80]:


topics = lda_model.describeTopics(maxTermsPerTopic=8)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)


# In[81]:


# Predictions
transformed = lda_model.transform(data_features)
print("Documents with their corresponding topic distribution:")
transformed.select("id", "topicDistribution").show(truncate=False)


# In[82]:


vocabulary = cv_model.vocabulary
for row in topics.collect():
    print("Topic {}: ".format(row['topic']))
    for termIndex in row['termIndices']:
        print(vocabulary[termIndex])
    print()


# ## EDA Conclusion

# In[83]:


df.printSchema()


# We have taken a close look at each of our non-text covariates independently and determined their relative utility for our sarcasm prediction task. Now it is time to feature engineer useful variables and remove any unnecessary or redundant ones.

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

# In[84]:


# Import preprocessing libraries
from pyspark.sql import functions as F


# In[85]:


# Add date-time variables
df2 = df.withColumn('month', F.month('created_utc'))            .withColumn('day_of_week', F.dayofweek('created_utc'))            .withColumn('hour', F.hour('created_utc'))

df2.show()


# In[86]:


# Calculate the number of nulls in each row by checking each column
null_check = df2.select([F.when(F.col(c).isNull(), 1).otherwise(0).alias(c) for c in df2.columns])

# Sum up the values across all columns for each row, resulting in a new DataFrame where each row has a sum of nulls
null_sums = null_check.withColumn('null_sum', sum(F.col(c) for c in null_check.columns))

# Filter to get only the rows with at least one null value and count them
num_rows_with_nulls = null_sums.filter(F.col('null_sum') > 0).count()

print(f"Number of rows with at least one null value: {num_rows_with_nulls}")


# In[87]:


num_rows_with_nulls/df2.count()


# In[88]:


df2.count()


# Since the amount of rows with missing values is less than 1%, let's filter out these rows.

# In[89]:


df3 = df2.dropna()
df3.count()


# In[90]:


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

df3.show()


# In[91]:


# Calculate the number of nulls in each row by checking each column
null_check = df3.select([F.when(F.col(c).isNull(), 1).otherwise(0).alias(c) for c in df3.columns])

# Sum up the values across all columns for each row, resulting in a new DataFrame where each row has a sum of nulls
null_sums = null_check.withColumn('null_sum', sum(F.col(c) for c in null_check.columns))

# Filter to get only the rows with at least one null value and count them
num_rows_with_nulls = null_sums.filter(F.col('null_sum') > 0).count()

print(f"Number of rows with at least one null value: {num_rows_with_nulls}")


# In[92]:


# Drop unnecessary columns
df3.columns


# In[93]:


clean_df = df3.select('label', 'comment', 'parent_comment', 'subreddit', 'score', 'month', 'day_of_week', 'hour', 'word_count', 'total_punctuation')
clean_df.show()


# ## Transformations
# Now that we have all the necessary features in our `clean_df`, we can start processing the data and performing transformations such as one-hot-encoding and maybe creating tf-idf vectors to input into our traditional machine learning models.

# In[94]:


# Subreddit value counts
clean_df.select('subreddit').distinct().count()


# In[95]:


# Step 1: Group by 'subreddit' and count the entries for each
subreddit_counts = clean_df.groupBy('subreddit').count()

# Step 2: Filter for subreddits with less than 5 comments
subreddits_less_than_5 = subreddit_counts.filter(col('count') < 2)

# Step 3: Count how many subreddits have less than 5 comments
number_of_subreddits_less_than_5 = subreddits_less_than_5.count()

print(f"Number of subreddits with less than 5 comments: {number_of_subreddits_less_than_5}")


# In[96]:


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
encoded_df.show()


# In[97]:


encoded_df.select('subredditVec').show()


# Vector Size (2718): This number represents the total number of unique subreddits (after processing, including the "other" category for subreddits with less than 2 comments) that have been identified across all comments in your dataset. It is the dimensionality of the one-hot encoded vector, meaning there are 2718 possible categories (subreddits) that each comment could belong to.
# 
# Index ([906], [3], [84], etc.): This number represents the index within the vector that corresponds to the specific subreddit a comment is associated with. The index starts at 0, so an index of 906 refers to the 907th subreddit in the sorted list of unique subreddits. Each comment's subreddit is represented by one of these indices, indicating which subreddit the comment belongs to.
# 
# Value ([1.0]): This indicates the value at the specified index. In the case of one-hot encoding, this will always be 1.0 for the index corresponding to the comment's subreddit, meaning the presence of that subreddit. All other positions in the vector will be 0 (not shown in the sparse vector representation), indicating the absence of those subreddits.

# In[98]:


# We can drop the subreddit feature
transformed_df = encoded_df.drop('subreddit')
transformed_df.show(5)


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

# In[99]:


# Now let's create tf-idf vectors for our text comments
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline

# Tokenize comment and parent_comment
tokenizer_comment = Tokenizer(inputCol="comment", outputCol="comment_tokens")
tokenizer_parent_comment = Tokenizer(inputCol="parent_comment", outputCol="parent_comment_tokens")

# Apply HashingTF
hashingTF_comment = HashingTF(inputCol="comment_tokens", outputCol="rawFeatures_comment", numFeatures=2**16)
hashingTF_parent_comment = HashingTF(inputCol="parent_comment_tokens", outputCol="rawFeatures_parent_comment", numFeatures=2**16)

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
tfidf_df.select("features_comment", "features_parent_comment").show(5)


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

# In[100]:


tfidf_df.show(5)


# In[101]:


tfidf_df.columns


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

# In[102]:


# Now we can drop the comment and parent comment since they are represented as tf-idf vectors
final_df = tfidf_df.select('features_comment', 'features_parent_comment', 'subredditVec', 'score', 'month', 'day_of_week', 'hour', 'word_count', 'total_punctuation', 'label')
final_df.columns


# In[103]:


final_df.show()


# Now, we can move onto scaling and splitting our data for modeling.

# # Scaling and Splitting

# In[104]:


from pyspark.ml.feature import VectorAssembler

# List of numerical columns to scale
numericCols = ['score', 'month', 'day_of_week', 'hour', 'word_count', 'total_punctuation']

# Assemble numerical features into a vector
assembler = VectorAssembler(inputCols=numericCols, outputCol="numeric_features")
final_df = assembler.transform(final_df)


# In[105]:


from pyspark.ml.feature import StandardScaler

# Scale the numerical features
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features", withStd=True, withMean=False)
scalerModel = scaler.fit(final_df)
final_df = scalerModel.transform(final_df)


# In[106]:


final_df.columns


# In[107]:


scaled_df = final_df.select('features_comment', 
                            'features_parent_comment', 
                            'subredditVec', 
                            'scaled_numeric_features', 
                            'label')
scaled_df.show(5)


# In[108]:


train_df, val_df = scaled_df.randomSplit([0.8, 0.2], seed=22)

train_rows = train_df.count()
train_cols = len(train_df.columns)

val_rows = val_df.count()
val_cols = len(val_df.columns)

print(f"Shape of train_df: ({train_rows}, {train_cols})")
print(f"Shape of val_df: ({val_rows}, {val_cols})")


# ### Train and Evaluate using only text covars

# In[109]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["features_comment", "features_parent_comment"], outputCol="features")

# Transform the dataset to include a new column 'features' that combines 'features_comment' and 'features_parent_comment'
combined_df = assembler.transform(scaled_df)

# Split the data into training and validation sets
train_df, val_df = combined_df.randomSplit([0.8, 0.2], seed=22)


# In[110]:


from pyspark.ml.classification import LogisticRegression

# Initialize the logistic regression model
lr = LogisticRegression(featuresCol='features', labelCol='label')

# Fit the model on the training data
lrModel = lr.fit(train_df)


# In[111]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

# Predict on the validation data
predictions = lrModel.transform(val_df)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy}")


# In[112]:


from pyspark.sql.types import FloatType

# Convert predictions and labels to float type
predictions = predictions.withColumn("label", predictions["label"].cast(FloatType()))
predictions = predictions.withColumn("prediction", predictions["prediction"].cast(FloatType()))

# Prepare the RDD required for MulticlassMetrics
predictionAndLabels = predictions.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Calculate precision, recall, and F1 Score
precision = metrics.precision(1.0)
recall = metrics.recall(1.0)
f1Score = metrics.fMeasure(1.0, beta=1.0)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1Score}")


# In[ ]:





# ### Naive Bayes using final_df

# In[117]:


from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder     .appName("NaiveBayesValidation")     .getOrCreate()

# Load the final DataFrame
#final_df = spark.read.csv("path_to_final_df.csv", header=True, inferSchema=True)

# Split the data into training and validation sets (80% training, 20% validation)
train_data, val_data = final_df.randomSplit([0.8, 0.2], seed=42)

# Initialize the Naive Bayes classifier
nb = NaiveBayes(featuresCol="features_comment", labelCol="label")

# Train the Naive Bayes model on the training data
nb_model = nb.fit(train_data)

# Make predictions on the validation data
val_predictions = nb_model.transform(val_data)

# Evaluate the predictions using accuracy as the metric
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
validation_accuracy = evaluator.evaluate(val_predictions)

# Print the validation accuracy
print("Validation Accuracy:", validation_accuracy)

# Stop the Spark session
#spark.stop()


# In[114]:


### Naive Bayes using scaled df
##doesn't work  tfidf_df


# In[115]:


from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder     .appName("NaiveBayesValidation")     .getOrCreate()

# Load the final DataFrame
#final_df = spark.read.csv("path_to_final_df.csv", header=True, inferSchema=True)

# Split the data into training and validation sets (80% training, 20% validation)
train_data, val_data = tfidf_df.randomSplit([0.8, 0.2], seed=42)

# Initialize the Naive Bayes classifier
nb = NaiveBayes(featuresCol="features_comment", labelCol="label")

# Train the Naive Bayes model on the training data
nb_model = nb.fit(train_data)

# Make predictions on the validation data
val_predictions = nb_model.transform(val_data)

# Evaluate the predictions using accuracy as the metric
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
validation_accuracy = evaluator.evaluate(val_predictions)

# Calculate precision, recall, and F1 score
true_positives = val_predictions.filter("label = 1 AND prediction = 1").count()
false_positives = val_predictions.filter("label = 0 AND prediction = 1").count()
true_negatives = val_predictions.filter("label = 0 AND prediction = 0").count()
false_negatives = val_predictions.filter("label = 1 AND prediction = 0").count()

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print evaluation metrics
print("Validation Accuracy:", validation_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

#spark.stop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




