# DS410 Final Project

## By: Team SLAMAC
## Adam Torres Encarnacion, Akshaya Shyamsundar Rekha, Corrina  Sigmund, Lauren Miller, Maya Nagiub, and Sneha Arya 

Our project aims to predict the sentiment of an Amazon review given its underlying text content.

The dataset is from a public Kaggle dataset (link: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews). The data is not synthetically generated; rather, it was sourced from the Stanford Network Analysis Project and spans dates from early 1995 to March 2013. Regarding the number of records, there are 4,000,000 total Amazon reviews which were balanced in terms of the review sentiment of positive and negative. On Kaggle, the reviews were split into balanced training and testing sets. The training set contained 3,600,000 reviews. The testing set contained 400,000 reviews. However, as we needed a train/validation/test split, we combined the training and testing sets from Kaggle into one large Amazon reviews dataset with 4,000,000 total reviews stored as a 1.6 GB csv. Additionally, we created a subset of ten percent of the total number of the reviews for running our code in local mode. This subset of reviews was balanced in terms of sentiment and contained 400,000 total reviews stored as a 166.2 MB csv.

In order to reproduce our local mode results:
1. Go to "Amazon_reviews_sample.csv.pdf" in this Github and use the link to download the Amazon_reviews_sample.csv
2. Go to "Final_Project_Local.ipynb" in this Github and download the Final_Project_Local.ipynb
3. Go the ICDS Roar Portal, create a Jupyter Lab session, and upload the Amazon_reviews_sample.csv and Final_Project_Local.ipynb to the work directory
4. Run all of the code cells in Final_Project_Local.ipynb in order

In order to reproduce our cluster mode results:
1. Go to "Amazon_reviews.csv.pdf" in this Github and use the link to download the Amazon_reviews.csv
2. Go to "Final_Project_Cluster.py" in this Github and download the Final_Project_Cluster.py
3. Go to "standalone.sh" in this Github and download the standalone.sh
4. Go the ICDS Roar Portal, create a Jupyter Lab session, and upload the Amazon_reviews.csv, Final_Project_Cluster.py, and standalone.sh to the work directory
5. Follow the instructions given by the professor in order to run the Final_Project_Cluster.py in cluster mode

NOTE - As the entire Final_Project_Cluster.py file takes a long time to run in totality, we recommend commenting out parts of the file and running them in batches.

