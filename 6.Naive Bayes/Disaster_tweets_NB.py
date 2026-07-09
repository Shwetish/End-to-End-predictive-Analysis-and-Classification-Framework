#====================================================================================
#Business Understanding: 
#====================================================================================
1. Business Problem Statement
Social media platforms (like Twitter) are the fastest way information spreads
during emergencies. However, these platforms are often flooded with a mix of 
actual emergency alerts and casual, metaphorical, or fake text (e.g., "This 
concert is fire!" or "I'm drowning in homework"). The problem is quickly and 
accurately distinguishing between real disaster reports and irrelevant or fake
tweets out of thousands of posts generated every second.

2. Business Objective
To build and implement a Naïve Bayes machine learning model that automatically
monitors incoming tweets and accurately classifies them into Real Disaster (1)
or Not a Real Disaster (0).

3. Motivation
Saves Lives: Emergency response teams, NGOs, and government agencies can 
discover real crises instantly without wasting precious time manually 
filtering through social media noise.

Reduces Misinformation: It prevents panic by stopping fake news or exaggerated
claims from being treated as real emergencies.

Resource Optimization: Helps dispatch rescue and relief teams to the exact 
locations where help is actually needed.

4. Constraints and Limitations
Linguistic Ambiguity: Words like "blaze", "fire", "bomb", or "wrecked" are 
frequently used as slang (e.g., "Our team wrecked the opposition"). The model 
might struggle with sarcasm or cultural context.

Data Quality (Noise): Tweets contain typos, emojis, URLs, and missing location
data (as seen in your screenshots), making text cleaning difficult.

Time Sensitivity: The system must process tweets in near real-time; a delay of
even a few minutes reduces the value of emergency data.

5. Success Criteria
Business Success Criteria
Timely Alerts: The system successfully flags real disasters fast enough for 
emergency response teams to act upon.

High Trust: Users and authorities can rely on the system without getting 
overwhelmed by false alarms.

ML (Machine Learning) Success Criteria
High Recall (Priority): We must minimize False Negatives (missing a real 
disaster). It is much better to accidentally flag a fake tweet as real than 
to miss a real earthquake or flood where lives are at risk.

High F1-Score: Since we need a balance between Precision (not flagging fake 
tweets) and Recall (catching all real tweets), aiming for a high F1-Score 
(e.g., >80%) is the ideal technical target.

#=============================================================================
#Data Understanding
#=============================================================================
'''
Name of Feature      Description                    Type of Variable        Relevance 
id              A unique identification number     Discrete / Nominal       Not 
                assigned to every single tweet row                         Relevant
                in the data file.                               
keyword         A specific, important word or 
                category extracted from the tweet       Categorical /      Highly 
                text (e.g., ablaze, crash, earthquake)   Nominal           Relevant
location        The physical region, city, or country 
                from where the user sent the tweet.     Categorical /      Low
                It contains a lot of blank rows.         Nominal         Relevance
text            The actual, raw message or caption
                posted by the user on the social        Unstructured     Critically
                network platform.                           Text           Relevant
target          The output column we want to predict.   Binary 
                1 = Real disaster tweet                 Categorical /     Output
                0 = Fake / Casual tweet                Target Variable    Variable
'''
# ==============================================================================
# Exploratory Data Analysis (EDA)- Disaster Tweets
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# -----------------------------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------------------------
# Replace with your actual local file path
tweets_df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/Disaster_tweets_NB .csv')

# -----------------------------------------------------------------------------
# Basic Understanding
# -----------------------------------------------------------------------------
tweets_df.head()
tweets_df.info()
tweets_df.describe(include='all')

# Inference:
# Dataset consists of text data (unstructured tweets) along with metadata 
# (keyword, location, and an internal sequential id).
# Target variable 'target' is a binary category (0 = Not a disaster, 1 = Real disaster) 
# which defines a text classification problem.

# -----------------------------------------
# Target Variable Distribution
# -----------------------------------------
print(tweets_df['target'].value_counts())
sns.countplot(x='target', data=tweets_df)
plt.show()

# Inference:
# Classes are relatively balanced but favor non-disaster tweets slightly (target=0).
# No extreme class imbalance is present, meaning default metrics like Accuracy 
# and F1-Score will be reliable during model evaluation.

#---------------------------------------------------------------
# Missing Value Analysis (Metadata Sparsity)
#------------------------------------------------------------
print(tweets_df[['keyword', 'location']].isnull().sum())

# Visualizing missing data patterns
sns.heatmap(tweets_df[['keyword', 'location']].isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Map (Keyword & Location)")
plt.show()

# Inference:
# 'location' contains a vast amount of missing values, making it highly sparse.
# 'keyword' has very few missing entries. 
# Due to structural text cleaning requirements, missing values in these metadata 
# columns must be handled before combining features or feeding them into an NLP pipeline.

#----------------------------------------------------------------------
# NLP Feature Engineering for Deeper Bivariate EDA
#----------------------------------------------------------------------
# Generating basic text metadata properties to understand characteristics
tweets_df['char_count'] = tweets_df['text'].astype(str).apply(len)
tweets_df['word_count'] = tweets_df['text'].astype(str).apply(lambda x: len(x.split()))

#--------------------------------------------------------------------
# Univariate Distribution - Character Count
#---------------------------------------------------------------------
sns.displot(tweets_df['char_count'], kde=True, color='purple')
plt.title("Distribution of Tweet Character Lengths")
plt.show()

# Inference:
# Character lengths show a sharp cutoff around 140 characters.
# This reflects the historical character limit constraint of the microblogging platform.

#-------------------------------------------------------------
# Bivariate Boxplot - Word Count vs Target
#------------------------------------------------------------
sns.boxplot(x='target', y='word_count', data=tweets_df, palette='Set1')
plt.title("Word Count vs Target Category")
plt.show()

# Inference:
# Tweets describing real disasters (target=1) tend to have a slightly higher 
# median word count and tighter distribution compared to non-disaster tweets.
# Informational emergency broadcast styles generally use more words to convey context.

#----------------------------------------------------------
# Top Keywords Exploration
#------------------------------------------------------------
top_keywords = tweets_df['keyword'].value_counts().head(15)
sns.barplot(y=top_keywords.index, x=top_keywords.values, palette='mako')
plt.title("Top 15 Most Frequent Keywords in Dataset")
plt.show()

# Inference:
# Certain terms like 'ablaze', 'wrecked', or 'earthquake' occur with high frequency.
# These explicit tokens will serve as strong conditional probabilities for the 
# Naïve Bayes classifier.

#-------------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS (Alternative Structural Length Overviews)
#-------------------------------------------------------------------------
# Updated to match your exact problem statement terminology
plt.hist(tweets_df[tweets_df['target']==1]['word_count'], alpha=0.5, label='Real Tweet (1)', color='red', bins=20)
plt.hist(tweets_df[tweets_df['target']==0]['word_count'], alpha=0.5, label='Fake Tweet (0)', color='blue', bins=20)
plt.title("Word Count Overlap Histogram: Real vs Fake Tweets")
plt.legend(loc='upper right')
plt.show()

#-----------------------------------------------------------------------
# Final EDA Conclusion
#------------------------------------------------------------------------
# This dataset represents a text-heavy predictive analytics classification problem.
# Standard numeric transformations (scaling) are not directly applicable here.
# Instead, text preprocessing (lowercasing, stopword removal, removing URLs/hashtags) 
# is strictly required.
# Bag-of-Words features via CountVectorizer or TF-IDF Vectorizer will be essential 
# to convert text strings into numerical sparse frequencies before fitting a 
#MultinomialNB model.

# ==============================================================================
# Data Preprocessing for Disaster Tweets Dataset 
# ==============================================================================
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix  # Added for safe memory handling

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/Disaster_tweets_NB .csv')
print("Initial Shape:", df.shape)

# -----------------------------------------------------------------------
# Missing Value Treatment
# -------------------------------------------------------------------------
cat_cols = ['keyword', 'location']
for col in cat_cols:
    df[col].fillna('unknown', inplace=True)

df['text'] = df['text'].fillna('')

# -----------------------------------------------------------------------
# Duplicate Removal
# --------------------------------------------------------------------------
df.drop_duplicates(subset=['text'], inplace=True)
print("After removing duplicates:", df.shape)

# -------------------------------------------------------------------------
# Text Cleaning & Normalization
# ----------------------------------------------------------------------------
def clean_tweet_text(text):
    text = str(text).lower() 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>+', '', text) 
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
    text = re.sub(r'\n', '', text) 
    text = re.sub(r'\w*\d\w*', '', text) 
    return text

df['clean_text'] = df['text'].apply(clean_tweet_text)
df['final_features'] = df['keyword'] + " " + df['clean_text']

# Format variables for modeling
X_raw = df['final_features']
y = df['target'].astype(int)

# ------------------------------------------------------------------------------
# Train-Test Split (Executed BEFORE vectorization to prevent data leakage)
# ------------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------------
# Vectorization & Feature Reductions (Fit on Train, Transform on Test)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
X_train_vectorized = vectorizer.fit_transform(X_train_raw)
X_test_vectorized = vectorizer.transform(X_test_raw)

selector = VarianceThreshold(threshold=0.0001)
X_train_scaled = selector.fit_transform(X_train_vectorized)
X_test_scaled = selector.transform(X_test_vectorized)

# ---------------------------------------------------------
# Class Imbalance Handling using SMOTE
# -------------------------------------------------------
print("Before SMOTE Train Balance:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
# Fix: Force dense handling back into compressed sparse matrices for model performance
X_train_res_dense, y_train_res = smote.fit_resample(X_train_scaled, y_train)
X_train_res = csr_matrix(X_train_res_dense)

print("After SMOTE Train Balance:\n", pd.Series(y_train_res).value_counts())

# Inference:
# Your final pipeline outputs X_train_res and y_train_res to train your model, 
# and evaluates performance safely against X_test_scaled and y_test.

# ==============================================================================
# Naive Bayes Model Development: Disaster Tweets Dataset
# ==============================================================================
import pandas as pd
import numpy as np
import re
import string
from scipy.sparse import csr_matrix  # Added for sparse memory handling

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import BernoulliNB  # Prioritized as per instructions
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------------------
# STEP 1: LOAD & RE-ESTABLISH CLEANED PREPROCESSED DATA
# ------------------------------------------------------------------------------
df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/Disaster_tweets_NB .csv')

df['keyword'] = df['keyword'].fillna('unknown')
df['text'] = df['text'].fillna('')
df.drop_duplicates(subset=['text'], inplace=True)

def clean_tweet_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_tweet_text)
df['final_features'] = df['keyword'] + " " + df['clean_text']

X_raw = df['final_features']
y = df['target']

# ------------------------------------------------------------------------------
# STEP 2: TRAIN-TEST SPLIT
# ------------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------------------
# 5.1 Build Model on Scaled Data (NLP Vector Scaling / Variance Reduction)
# ------------------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, min_df=2)

X_train_vectorized = vectorizer.fit_transform(X_train_raw)
X_test_vectorized = vectorizer.transform(X_test_raw)

selector = VarianceThreshold(threshold=0.0001)
X_train_scaled = selector.fit_transform(X_train_vectorized)
X_test_scaled = selector.transform(X_test_vectorized)

# Fix: Force SMOTE to process cleanly and convert back to sparse matrix to prevent memory crash
smote = SMOTE(random_state=42)
X_train_res_dense, y_train_res = smote.fit_resample(X_train_scaled, y_train)
X_train_res = csr_matrix(X_train_res_dense)  # Keeps model training fast and safe

# ------------------------------------------------------------------------------
# 5.2 Build Bernoulli Naïve Bayes Model 
# ------------------------------------------------------------------------------
bnb_model = BernoulliNB()
bnb_model.fit(X_train_res, y_train_res)

# ------------------------------------------------------------------------------
# 5.3 Validate the baseline model with test data
# ------------------------------------------------------------------------------
y_pred_bnb = bnb_model.predict(X_test_scaled)

print("--- BASELINE BERNOULLI NAIVE BAYES PERFORMANCE ---")
print("Test Accuracy :", accuracy_score(y_test, y_pred_bnb))
print("Precision     :", precision_score(y_test, y_pred_bnb))
print("Recall        :", recall_score(y_test, y_pred_bnb))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_bnb))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_bnb))

# ------------------------------------------------------------------------------
# 5.4 Tune the model and improve the accuracy
# ------------------------------------------------------------------------------
# Tweaking alpha parameter for Bernoulli distribution model
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
grid_search = GridSearchCV(BernoulliNB(), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

best_tweet_model = grid_search.best_estimator_
y_test_pred_tuned = best_tweet_model.predict(X_test_scaled)

print("\n--- TUNED BERNOULLI NAIVE BAYES PERFORMANCE ---")
print("Best Tuning Hyperparameter (alpha):", grid_search.best_params_)
print("Tuned Test Accuracy :", accuracy_score(y_test, y_test_pred_tuned))
print("Tuned Precision     :", precision_score(y_test, y_test_pred_tuned))
print("Tuned Recall        :", recall_score(y_test, y_test_pred_tuned))
print("\nTuned Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred_tuned))
#====================================================================================
6. Benefits and Business Impact of the Solution
#====================================================================================
Deploying an automated, real-time classifier for disaster tweets yields critical 
strategic advantages for humanitarian, news, and public safety organizations:

Accelerated Dispatch for Emergency First Responders: During major crises, response
 teams cannot manually read through millions of raw social media updates. This model 
 filters out casual commentary, letting emergency personnel instantly track actual
 distress reports, coordinate resources, and locate hot zones faster.

Proactive Brand Protection and Risk Management: Public infrastructure firms,
 utilities, and transport operators can use this classifier as an early-warning
 monitor. Identifying real-time network disruptions or structural safety hazards on
 social media allows companies to deploy field technicians and issue public updates
 before problems escalate.

Automated Data Filtering for News Desks and NGOs: Media organizations and relief 
groups can isolate breaking crisis event tickers from general social chatter 
automatically. This improves field reporting, reduces operational overhead, and
 speeds up structural mobilization during global emergencies.

Instant, High-Throughput Stream Inferences: Thanks to the low computational overhead
 of Naïve Bayes math, the model processes large text feeds with minimal hardware 
 resources. It can easily handle high-volume live data streams, making it a reliable
 production asset for emergency dashboard applications.