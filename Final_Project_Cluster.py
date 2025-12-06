#!/usr/bin/env python
# coding: utf-8

# ## DS 410 Final Project

# Importing libraries, Creating Spark Session, and Reading in the Dataset

# In[1]:


import pyspark
from pyspark import SparkContext
from pyspark.ml.classification import GBTClassifier
from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.sql.functions import lower
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
import xgboost
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import sparknlp
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt', quiet=True)
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.functions import vector_to_array
import seaborn as sns
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np
from sklearn.metrics import roc_curve, average_precision_score, f1_score, accuracy_score
import random
from pyspark.sql.functions import concat_ws, array_join
from pyspark.sql.functions import col, when, lower, split, array_remove, regexp_replace
from pyspark.ml.feature import StopWordsRemover
import matplotlib.pyplot as plt

# In[3]:


ss=(SparkSession.builder.appName("Final_Project").config("spark.executor.memory", "8g").config("spark.executor.cores", "2").config("spark.driver.memory", "8g").getOrCreate())


# In[4]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[12]:


schema = StructType([StructField("Sentiment", IntegerType(), nullable = False),                      StructField("Title", StringType(), nullable = False),                      StructField("Text", StringType(), nullable = False)
                    ])


# In[13]:


reviews1 = ss.read.csv("Amazon_reviews.csv", schema = schema, header = True, inferSchema = False)


# In[14]:



# In[15]:




# In[16]:


reviews = reviews1.na.drop()


# In[17]:


# reviews.printSchema()


# In[18]:


# reviews.show(5)


# ## EDA

# ## Distribution of Review Sentiment

# In[19]:

'''
reviews_label = reviews.withColumn("Labeled_Sentiment", when(reviews["Sentiment"] == 2, "Positive").when(reviews["Sentiment"] == 1, "Negative"))
sentiment_counts = reviews_label.groupBy("Labeled_Sentiment").count()
pandas_sentiment=sentiment_counts.toPandas()
plt.bar(pandas_sentiment["Labeled_Sentiment"],pandas_sentiment["count"])
plt.title("Distribution of Review Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches='tight')
# plt.show()


# Removing special characters

# In[12]:


from pyspark.sql.functions import regexp_replace
reviews_special_title = reviews_label.withColumn("Title", regexp_replace(col("Title"), "[^a-zA-Z0-9\\s]", ""))
reviews_only_text = reviews_special_title.withColumn("Text", regexp_replace(col("Text"), "[^a-zA-Z0-9\\s]", ""))



# Making all words lowercase and splitting by space

# In[13]:


reviews_split_title = reviews_only_text.withColumn("Title", split(lower(reviews_only_text["Title"]), " "))
reviews_split_title_text = reviews_split_title.withColumn("Text", split(lower(reviews_split_title["Text"]), " "))
reviews_split_title_text = (reviews_split_title_text.withColumn("Title", array_remove("Title", "")).withColumn("Text",  array_remove("Text", "")))


# Removing stop words, making rdd of title of reviews and text of reviews for positive and negative sentiment

# In[14]:


remover = StopWordsRemover(inputCols=["Title", "Text"], outputCols=["Title_No_Stop", "Text_No_Stop"])
reviews_no_stop_words = remover.transform(reviews_split_title_text)
reviews_combined = reviews_no_stop_words.withColumn("Combined", concat_ws(" ", "Title_No_Stop", "Text_No_Stop"))
reviews_pos = reviews_combined.filter(col("Labeled_Sentiment") == "Positive").select("Combined").rdd.flatMap(lambda row: row["Combined"].split())
reviews_neg = reviews_combined.filter(col("Labeled_Sentiment") == "Negative").select("Combined").rdd.flatMap(lambda row: row["Combined"].split())

# Concatenate title and reviews rdd for positive and negative sentiment

# In[15]:


# Map words to key-value pairs ex. (word, 1)

# In[16]:


reviews_pos_key_val = reviews_pos.map(lambda x: (x, 1))
reviews_neg_key_val = reviews_neg.map(lambda x: (x, 1))


# Reduce key-value pairs by key (aggregate counts of words)

# In[17]:


reviews_pos_key_val_reduced = reviews_pos_key_val.reduceByKey(lambda x, y: x + y, 4)
reviews_neg_key_val_reduced = reviews_neg_key_val.reduceByKey(lambda x, y: x + y, 4)


# Sort aggregated counts of words in descending order

# In[18]:


reviews_pos_key_val_sorted = reviews_pos_key_val_reduced.sortBy(lambda pair: pair[1], ascending=False)
reviews_neg_key_val_sorted = reviews_neg_key_val_reduced.sortBy(lambda pair: pair[1], ascending=False)


# Get ten most frequent words for title of reviews and text of reviews

# In[19]:


reviews_pos_key_val_sorted_top_ten = reviews_pos_key_val_sorted.take(10)
reviews_neg_key_val_sorted_top_ten  = reviews_neg_key_val_sorted.take(10)


# --- Positive words ---
plt.figure(figsize=(8, 4))  # start a fresh figure
plt.bar(
    [item[0] for item in reviews_pos_key_val_sorted_top_ten],
    [item[1] for item in reviews_pos_key_val_sorted_top_ten]
)
plt.xlabel("Word")
plt.ylabel("Number of Occurrences")
plt.title("Top Ten Most Frequently Used Words in Positive Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("word_freq_pos_distribution.png", dpi=300, bbox_inches='tight')
plt.close()  # close this figure so it doesn't bleed into the next one

# --- Negative words ---
plt.figure(figsize=(8, 4))  # new figure again
plt.bar(
    [item[0] for item in reviews_neg_key_val_sorted_top_ten],
    [item[1] for item in reviews_neg_key_val_sorted_top_ten]
)
plt.xlabel("Word")
plt.ylabel("Number of Occurrences")
plt.title("Top Ten Most Frequently Used Words in Negative Reviews")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("word_freq_neg_distribution.png", dpi=300, bbox_inches='tight')
plt.close()


# # PreProcessing Data

# Create a general stemming function

# In[22]:


#General function to stem words
def stem_words(words):
    #if there are no words, return an empty list
    if words is None:
        return []
    
    #define a porter stemmer object
    stemmer = PorterStemmer()
    cleaned_stemmed = []
    
    #iterate through all words in review
    for word in words:
        #apply the stemmer object
        stemmed = stemmer.stem(word.lower())
        #add to list of stemmed words
        cleaned_stemmed.append(stemmed)
    
    #return stemmed list
    return cleaned_stemmed

#create spark user defined function for stem_words
stem_udf = udf(stem_words, ArrayType(StringType()))


# Apply stemming function to text of reviews

# In[23]:


reviews_stemmed = reviews_no_stop_words.withColumn(
    "Title_Stemmed",
    stem_udf(col("Title_No_Stop"))
).withColumn(
    "Text_Stemmed",
    stem_udf(col("Text_No_Stop"))
)

# Preparing Data for Hugging Face

# In[24]:


#Combined stemmed title and text of reviews into one column called text
#Create binary labels where zero is negative sentiment and one is positive sentiment
reviews_for_transformers = reviews_stemmed.withColumn(
    "text",
    concat_ws(" ", 
              array_join(col("Title_Stemmed"), " "),
              array_join(col("Text_Stemmed"), " "))
).withColumn(
    "label",  # Binary label: 0=Negative, 1=Positive
    when(col("Sentiment") == 2, 1).otherwise(0)
).select("text", "label", "Sentiment", "Labeled_Sentiment")



# Create Train/Validation/Test Split with a 60/20/20 Split

# In[25]:



#Split data into train/validation/test set with 60/20/20 split, set random seed for reproduciblity
train_data, valid_data, test_data = reviews_for_transformers.randomSplit([0.6, 0.2, 0.2], seed=42)

#Cache the three data sets
train_data.cache()
valid_data.cache()
test_data.cache()

#Count the records to force computation and caching
train_count = train_data.count()
valid_count = valid_data.count()
test_count = test_data.count()

# print("Total Number of Reviews in Training Set: ", train_count)
# print("Total Number of Reviews in Validation Set: ", valid_count)
# print("Total Number of Reviews in Test Set: ", test_count)


# Save PreProcessed Data to be Accessed later

# In[26]:


#Define output path to save preprocessed data to
OUTPUT_PATH = "preprocessed_data_cluster_amazon"

#Save preprocessed tranining/validation/test data, overwritting what was there before (avoid overwriting errors)
train_data.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/train_transformer_ready")
valid_data.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/valid_transformer_ready")
test_data.write.mode("overwrite").parquet(f"{OUTPUT_PATH}/test_transformer_ready")

#Unpersist in order to free up memory
train_data.unpersist()
valid_data.unpersist()
test_data.unpersist()

#The data can be loaded with
#train_df = spark.read.parquet("preprocessed_data_sample/train_transformer_ready")
#valid_df = ss.read.parquet("preprocessed_data_sample/valid_transformer_ready")
#test_df  = ss.read.parquet("preprocessed_data_sample/test_transformer_ready")

# # Model Building

# Data PreProcessing for Model Building

# In[27]:
'''

#read in datasets from saved folder
train_df = ss.read.parquet("preprocessed_data_cluster_amazon/train_transformer_ready")
valid_df = ss.read.parquet("preprocessed_data_cluster_amazon/valid_transformer_ready")
test_df  = ss.read.parquet("preprocessed_data_cluster_amazon/test_transformer_ready")
#cast the labels into doubles, which is necessary for later models
for name in ("train_df","valid_df","test_df"):
    locals()[name] = locals()[name].withColumn("label", col("label").cast("double"))
#create tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
#create hashing object to transform tokenized text into 50,000 term frequency vector
tf  = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=50_000)
#create inverse document frequency transformer
idf = IDF(inputCol="rawFeatures", outputCol="features")
#create pipeline of data processing with review text being tokenized, transformed into term frequncy vector, then TF-IDF features, fit the preprocessing training data
featurizer = Pipeline(stages=[tokenizer, tf, idf]).fit(train_df)
#transform the traning/validation/test data, repartitioning the resulting dataframe and saving it in memory

train_f = featurizer.transform(train_df) \
    .select("features", "label") \
    .repartition(64) \
    .persist(StorageLevel.MEMORY_AND_DISK)

valid_f = featurizer.transform(valid_df) \
    .select("features", "label") \
    .repartition(32) \
    .persist(StorageLevel.MEMORY_AND_DISK)

test_f  = featurizer.transform(test_df) \
    .select("features", "label") \
    .repartition(32) \
    .persist(StorageLevel.MEMORY_AND_DISK)

# Force materialization so caching actually happens
train_f.count()
valid_f.count()
test_f.count()


# Creating a general model training and evaluation function

# In[31]:


#General model training function
def evaluate_model(model, model_name):
    # fit model on training data
    model = model.fit(train_f)

    # predict label on validation data
    valid_pred = model.transform(valid_f)

    # create evaluators for AUC-PR, AUC-ROC, F1, and Accuracy
    e_aucpr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
    e_auc   = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    e_f1    = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    e_acc   = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

       # ---- Evaluate metrics ----
    auc_pr  = e_aucpr.evaluate(valid_pred)
    auc_roc = e_auc.evaluate(valid_pred)
    f1      = e_f1.evaluate(valid_pred)
    acc     = e_acc.evaluate(valid_pred)

    # ---- Write metrics to text file ----
    metrics_filename = f"{model_name}_metrics.txt"
    with open(metrics_filename, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("----------------------------\n")
        f.write(f"AUC-PR:  {auc_pr}\n")
        f.write(f"AUC-ROC: {auc_roc}\n")
        f.write(f"F1:       {f1}\n")
        f.write(f"Accuracy: {acc}\n")


    # get raw decision score from model
    valid_scores = valid_pred.withColumn("raw_score", vector_to_array("rawPrediction")[0])

    # convert to pandas
    pdf = valid_scores.select("label", "raw_score").toPandas()

    # ROC metrics
    fpr, tpr, _ = roc_curve(pdf["label"], pdf["raw_score"])

    # ROC plot
    plt.figure(figsize=(7, 6))
    plt.plot(tpr, fpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Validation ROC Curve")
    plt.grid(True)
    plt.savefig(f"{model_name}_ROC.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # confusion matrix
    preds_and_labels = valid_pred.select(['prediction', 'label']).rdd.map(tuple)
    metrics = MulticlassMetrics(preds_and_labels)
    conf_matrix = metrics.confusionMatrix().toArray()
    labels = sorted(valid_pred.select('label').distinct().rdd.flatMap(lambda x: x).collect())
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    plt.savefig(f"{model_name}_confusion.png", dpi=300, bbox_inches='tight')
    # plt.show()


# # SVM

# In[32]:
'''

#define Linear SVC model
svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1)
#call evaluate_model function
evaluate_model(svm, "SVM")


# # Decision Tree

# In[45]:


#define decision tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=10)
#call evaluate_model function
evaluate_model(dt, "Decision_Tree")

# # Random Forest

# In[ ]:


#define random forest model
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,        
    maxDepth=5,        
    featureSubsetStrategy="auto",
)
#call evalaute_model function
evaluate_model(rf, "Random_Forest")

# # Naive Bayes

# In[ ]:


#define naive bayes model
nb = NaiveBayes(
    labelCol="label",
    featuresCol="features",
    modelType="multinomial",
    smoothing=1.0
)

#call evaluate_model function
evaluate_model(nb, "Naive_Bayes")


# # XG Boost

# In[ ]:


gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    maxDepth=5
)

evaluate_model(gbt, "GBT")
'''

# # Hyperparameter Tuning on Linear SVC

# In[29]:


#define linear SVC model
svm_base = LinearSVC(featuresCol="features", labelCol="label")
#define want to train 50 randomly selected model
n_samples = 50
#set random seeds
np.random.seed(42)
random.seed(42)

param_grid = []
#loop through n_samples to randomly select param_grid for each iteration
for _ in range(n_samples):
    #defined parameters grid to sample from
    param_grid.append({
        "regParam": 10**np.random.uniform(-4, 0),
        "maxIter": np.random.randint(1, 21) * 10,
        "tol": 10**np.random.uniform(-6, -2),
        "fitIntercept": bool(np.random.choice([True, False])),
        "standardization": bool(np.random.choice([True, False])),
    })
    
#define AUC-ROC evaluator as going to use AUC-ROC to choose the best hyperparameter tuned model
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

results = []
#loop over set of randomly selected parameter settings
for params in param_grid:
    #get copy of svm_base model
    svm = svm_base.copy()
    #set paramters to linear SVC model defined above
    svm = svm.setParams(**params)
    #fit model on training set
    model = svm.fit(train_f)
    #predict on validation set
    pred = model.transform(valid_f)
    #compute AUC-ROC
    auc = evaluator_auc.evaluate(pred)
    #append parameters and AUC-ROC score to results
    results.append({
        "params": params,
        "AUC_ROC": auc
    })

#get best result by AUC-ROC
best_result = max(results, key=lambda x: x["AUC_ROC"])
#output best hyperparameters




with open("best_hyperparameters.txt", "w") as f:
    f.write("Best hyperparameters:\n")
    for name, value in best_result["params"].items():
        f.write(f"{name}: {value}\n")
#get paramters of best model
best_params = best_result["params"]
#get copy of base svc model
svm = svm_base.copy()
#set best hyperparamters
svm = svm.setParams(**best_params)
#fit best model on training set
best_model = svm.fit(train_f)
#transform validation set with best model
valid_pred = best_model.transform(valid_f)
#create evaluator objects
e_aucpr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
e_f1    = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
e_acc   = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")


#Evaluate metrics on validation data set
auc_pr = e_aucpr.evaluate(valid_pred)
auc_roc = best_result["AUC_ROC"]   # or evaluator_auc.evaluate(valid_pred) if you prefer
f1 = e_f1.evaluate(valid_pred)
acc = e_acc.evaluate(valid_pred)
# ---- Write to file ----
with open("best_model_metrics.txt", "w") as f:
    f.write("Best Model Evaluation Metrics\n")
    f.write("-----------------------------\n")
    f.write(f"AUC-PR: {auc_pr}\n")
    f.write(f"AUC-ROC: {auc_roc}\n")
    f.write(f"F1: {f1}\n")
    f.write(f"Accuracy: {acc}\n")
valid_pred_scores = valid_pred.withColumn("raw_score", vector_to_array("rawPrediction")[0])
#convert predictions to pandas
pdf_valid = valid_pred_scores.select("label", "raw_score").toPandas()
#compute ROC curve
fpr, tpr, _ = roc_curve(pdf_valid["label"], pdf_valid["raw_score"])
#create ROC curve
plt.figure(figsize=(7, 6))
plt.plot(tpr, fpr, linewidth=2, label=f"AUC-ROC = {best_result['AUC_ROC']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve: Best LinearSVC Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f"tuned_confusion.png", dpi=300, bbox_inches='tight')
# plt.show()

#get prediction and acutal labels as tuple rdd in form (prediction, label)
preds_and_labels = valid_pred.select(['prediction', 'label']).rdd.map(tuple)
#create metrics object
metrics = MulticlassMetrics(preds_and_labels)
#create confusion matrix
conf_matrix = metrics.confusionMatrix().toArray()
#get valid labels
labels = sorted(valid_pred.select('label').distinct().rdd.flatMap(lambda x: x).collect())
#create pandas data frame of confusion matrix
df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
#create confusion matrix heat map
plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Best Model Validation Set Confusion Matrix')
plt.savefig(f"tuned_ROC.png", dpi=300, bbox_inches='tight')
# plt.show()


# # Evaluation on Test Split

# In[30]:


#use best model to predict on test set
test_pred = best_model.transform(test_f)
#create evaluator objects
e_aucpr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
e_auc   = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
e_f1    = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
e_acc   = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
#Evaluate metrics on test data set
auc_pr = e_aucpr.evaluate(test_pred)
auc_roc = e_auc.evaluate(test_pred)
f1 = e_f1.evaluate(test_pred)
acc = e_acc.evaluate(test_pred)
with open("best_test_model_metrics_test.txt", "w") as f:
    f.write("Best Model Evaluation Metrics\n")
    f.write("-----------------------------\n")
    f.write(f"AUC-PR: {auc_pr}\n")
    f.write(f"AUC-ROC: {auc_roc}\n")
    f.write(f"F1: {f1}\n")
    f.write(f"Accuracy: {acc}\n")
#get raw decision score from model
test_scores = test_pred.withColumn("raw_score", vector_to_array("rawPrediction")[0])
#convert to pandas data frame
pdf = test_scores.select("label", "raw_score").toPandas()
#get roc metrics
fpr, tpr, _ = roc_curve(pdf["label"], pdf["raw_score"])
#create ROC plot
plt.figure(figsize=(7, 6))
plt.plot(tpr, fpr, linewidth=2, label=f"AUC-ROC = {e_auc.evaluate(test_pred)}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve: Best Model Test Set")
plt.grid(True)
# plt.show()
plt.savefig(f"test_tuned_confusion.png", dpi=300, bbox_inches='tight')

#get prediction and acutal labels as tuple rdd in form (prediction, label)
preds_and_labels = test_pred.select(['prediction', 'label']).rdd.map(tuple)
#create metrics object
metrics = MulticlassMetrics(preds_and_labels)
#create confusion matrix
conf_matrix = metrics.confusionMatrix().toArray()
#get valid labels
labels = sorted(valid_pred.select('label').distinct().rdd.flatMap(lambda x: x).collect())
#create pandas data frame of confusion matrix
df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
#create confusion matrix heat map
plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Best Model Test Set Confusion Matrix')
plt.savefig(f"test_tuned_ROC.png", dpi=300, bbox_inches='tight')
# plt.show()

# In[ ]:



ss.stop()

