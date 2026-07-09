#==============================================================================================
#Data Understanding :-
#==============================================================================================
'''
Name of Feature                Description               Type of Variable             Relevance 
type            The output column we want to predict.    Binary Categorical      Output Variable
                spam = Fraudulent / advertisement message  / Target Variable
                ham = Normal / legitimate text                                 
text            The actual, raw message or SMS text         Unstructured        Critically
                sent by the user.                                  Text         Relevant

'''


# ==============================================================================
# Exploratory Data Analysis (EDA) with Inline Inference - SMS Spam
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# -----------------------------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------------------------
# Add encoding='latin-1' inside the read_csv function
sms_df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/sms_raw_NB .csv', encoding='latin-1')

# Verify it loaded correctly
print("Dataset Shape:", sms_df.shape)
sms_df.head()

# -----------------------------------------------------------------------------
# Basic Understanding
# -----------------------------------------------------------------------------
sms_df.head()
sms_df.info()
sms_df.describe(include='all')

# Inference:
# Dataset consists of two columns: 'type' (the class label) and 'text' (the raw SMS string).
# Target variable 'type' is categorical with two distinct classes ('ham' and 'spam').
# This is a classical binary text classification problem.

# -----------------------------------------
# Target Variable Distribution
# -----------------------------------------
print(sms_df['type'].value_counts())
sns.countplot(x='type', data=sms_df, palette='Set1')
plt.show()

# Inference:
# The dataset exhibits a severe class imbalance. 
# The vast majority of messages are 'ham' (legitimate), while 'spam' forms a small minority.
# This severe imbalance must be accounted for during evaluation using Precision, Recall, and F1-Score.

#---------------------------------------------------------------
# Missing Value Analysis
#------------------------------------------------------------
print("Missing values in dataset:\n", sms_df.isnull().sum())

# Inference:
# There are no missing values in either the target 'type' or the 'text' columns.
# The data is structurally complete, meaning we do not need imputation strategies before tokenization.

#----------------------------------------------------------------------
# NLP Feature Engineering for Deeper Bivariate EDA
#----------------------------------------------------------------------
# Extracting structural properties from text to evaluate statistical patterns
sms_df['char_count'] = sms_df['text'].astype(str).apply(len)
sms_df['word_count'] = sms_df['text'].astype(str).apply(lambda x: len(x.split()))
sms_df['has_currency_symbol'] = sms_df['text'].apply(lambda x: int(any(sym in str(x) for sym in ['£', '$', '€'])))

#--------------------------------------------------------------------
# Univariate Distribution - Character Count
#---------------------------------------------------------------------
sns.displot(sms_df['char_count'], kde=True, color='darkgreen')
plt.title("Distribution of SMS Character Lengths")
plt.show()

# Inference:
# The distribution of character length shows a bimodal shape with a heavy concentration 
# under 100 characters, reflecting the rapid, casual nature of instant messaging.

#-------------------------------------------------------------
# Bivariate Boxplot - Character Count vs Target Type
#------------------------------------------------------------
sns.boxplot(x='type', y='char_count', data=sms_df, palette='coolwarm')
plt.title("Message Character Length by SMS Category")
plt.show()

# Inference:
# A massive behavioral difference is visible here. 
# 'Spam' messages consistently display significantly higher character lengths (tightly grouped around 130-160 characters).
# 'Ham' messages are highly variable but generally much shorter, often capturing quick conversational fragments.

#----------------------------------------------------------
# Bivariate Analysis - Specialized Tokens (Currency Triggers)
#------------------------------------------------------------
sns.countplot(x='has_currency_symbol', hue='type', data=sms_df, palette='viridis')
plt.title("Presence of Currency Symbols vs Message Type")
plt.show()

# Inference:
# Financial tokens (like '£' or '$') are disproportionately present in 'spam' messages.
# These terms act as high-probability indicators that the Multinomial Naïve Bayes model can exploit during training.


#-------------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS (Word Count Densities)
#-------------------------------------------------------------------------
plt.hist(sms_df[sms_df['type']=='spam']['word_count'], alpha=0.6, label='Spam', color='red', bins=25)
plt.hist(sms_df[sms_df['type']=='ham']['word_count'], alpha=0.4, label='Ham', color='blue', bins=25)
plt.title("Word Count Variations Across Ham and Spam Classes")
plt.xlim(0, 60)
plt.legend()
plt.show()

#-----------------------------------------------------------------------
# Final EDA Conclusion
#------------------------------------------------------------------------
# The dataset presents a highly text-dependent, imbalanced classification objective.
# Standard distance metrics and numerical scaling do not apply to this raw text array.
# Text preprocessing (lowercasing, punctuation stripping, digit identification) is mandatory.
# Feature extraction via CountVectorizer or TfidfVectorizer is required to represent word frequencies.
# Due to the categorical text nature, a Multinomial Naïve Bayes classifier is mathematically ideal here.

# ==============================================================================
# Data Preprocessing for SMS Spam Dataset (sms_raw_NB) - Fully Corrected
# ==============================================================================
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------------------------
# Added encoding='latin-1' to safely parse unusual characters and promotional marks
sms_df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/sms_raw_NB .csv', encoding='latin-1')

print("Initial Dataset Shape:", sms_df.shape)
sms_df.head()

# Data Types Check
sms_df.info()

# Inference:
# Dataset is strictly composed of a textual sequence column ('text') and a category column ('type').
# Target variable 'type' represents binary text classification ('ham' or 'spam').

# ------------------------------------------------------------------------------
# Missing Value Analysis & Treatment
# ------------------------------------------------------------------------------
print("Missing values:\n", sms_df.isnull().sum())

# Safeguarding text column matrix against unexpected float NaN occurrences
sms_df['text'] = sms_df['text'].fillna('')

# Inference:
# No missing values are present in this data pull. 
# Explicit empty string conversion ensures text parsing routines do not crash.

# ------------------------------------------------------------------------------
# Duplicate Removal
# ------------------------------------------------------------------------------
sms_df.drop_duplicates(subset=['text'], inplace=True)
print("After removing duplicates:", sms_df.shape)

# Inference:
# Text systems frequently contain identical spam broadcast blasts or template texts.
# Dropping duplicates protects validation partitions from processing leakage.

# ------------------------------------------------------------------------------
# Outlier Detection via Text Length Analysis
# ------------------------------------------------------------------------------
# In NLP workflows, text length metrics replace standard numeric attribute boxplots
sms_df['text_len'] = sms_df['text'].astype(str).apply(len)

sns.boxplot(x=sms_df['text_len'])
plt.title("SMS Character Length Distribution (Outlier Assessment)")
plt.show()

# Inference:
# Long text messages create a heavily right-skewed length boxplot profile.
# These represent long, detailed messages, distinct from rapid conversational bursts.

# ------------------------------------------------------------------------------
# Text Cleaning & Normalization
# ------------------------------------------------------------------------------
def clean_sms_text(text):
    text = str(text).lower()                                         # Lowercase conversion
    text = re.sub(r'https?://\S+|www\.\S+', '', text)                # Remove web URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Strip punctuation marks
    text = re.sub(r'\n', '', text)                                   # Remove text newlines
    text = re.sub(r'\w*\d\w*', ' numtoken ', text)                   # Replace numbers/codes with standard placeholder
    return text

sms_df['clean_text'] = sms_df['text'].apply(clean_sms_text)

# Inference:
# Converting numeric text anomalies (like phone numbers, cash reward codes) to 'numtoken' 
# captures structural spam flags without generating thousands of unique, arbitrary numbers.

# ------------------------------------------------------------------------------
# Encoding Target Variable
# ------------------------------------------------------------------------------
# Clean any accidental spaces from column headers before mapping
sms_df.columns = sms_df.columns.str.strip()

# Map string labels into discrete structural integers
sms_df['type'] = sms_df['type'].map({'ham': 0, 'spam': 1})

# Inference:
# Binary mapping confirmed: 0 -> Ham (Legitimate), 1 -> Spam.
# This conforms to standard classification evaluation arrays.

# ------------------------------------------------------------------------------
# Feature Isolation & Memory Management
# ------------------------------------------------------------------------------
X_raw = sms_df['clean_text']
y = sms_df['type']

# Drop intermediate analysis data markers to clear working memory
sms_df.drop(columns=['text', 'text_len', 'clean_text'], inplace=True, errors='ignore')

# Inference:
# Dropping unused feature remnants prevents tracking inflation across local dependencies.

# ------------------------------------------------------------------------------
# Train-Test Split (Executed BEFORE vectorization to completely prevent leakage)
# ------------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# Inference:
# Given the high proportion of legitimate messages, a stratified split is critical 
# to keep the train and test sets balanced with the same spam ratio.

# ------------------------------------------------------------------------------
# Vectorization & Zero Variance Feature Removal
# ------------------------------------------------------------------------------
# Transform processed text strings into weighted token occurrence structures
vectorizer = TfidfVectorizer(max_features=3000, min_df=3)
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)

# Weed out tokens displaying static behavior patterns across rows
selector = VarianceThreshold(threshold=0.0001)
X_train_final = selector.fit_transform(X_train_vec)
X_test_final = selector.transform(X_test_vec)

# Inference:
# TF-IDF changes conversational string fragments into a math-ready numerical sparse grid.
# Removing columns with near-zero variance eliminates terms that show up too rarely to have predictive weight.

# ------------------------------------------------------------------------------
# Feature Scaling (NLP Matrix Transformation)
# ------------------------------------------------------------------------------
# Note: X_train_final and X_test_final serve as your scaled inputs.
# TF-IDF operates within an implicitly standardized range. 
# Standard Scaler is intentionally skipped to preserve matrix sparsity and prevent negative values.

# Inference:
# The mathematical range of word frequencies remains naturally bounded between 0 and 1.

# ------------------------------------------------------------------------------
# Class Imbalance Handling using SMOTE
# ------------------------------------------------------------------------------
print("Before SMOTE Training Balance:\n", y_train.value_counts())

# Generate synthetic rows and wrap back into a compressed sparse matrix format safely
smote = SMOTE(random_state=42)
X_train_res_dense, y_train_res = smote.fit_resample(X_train_final, y_train)
X_train_res = csr_matrix(X_train_res_dense)

print("After SMOTE Training Balance:\n", pd.Series(y_train_res).value_counts())

# Inference:
# SMOTE resolves the massive class imbalance by oversampling sparse spam indicators.
# This prevents the Multinomial Naïve Bayes model from safely defaulting to '0' (ham) for every prediction.
# ==============================================================================
# Naive Bayes Model Development: SMS Spam/Ham Dataset (sms_raw_NB)
# ==============================================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------------------
# STEP 1: LOAD DATASET & BASIC SEPARATION
# ------------------------------------------------------------------------------
# 1. Load the dataset into sms_df
sms_df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/sms_raw_NB .csv', encoding='latin-1')

# Print the shape to make sure it worked perfectly
print("Dataset successfully loaded! Shape:", sms_df.shape)

# 2. Clean up any accidental blank spaces in the column headers
sms_df.columns = sms_df.columns.str.strip()

# 3. Drop rows with missing text values using sms_df
sms_df.dropna(subset=['text'], inplace=True)

# 4. Encode label strings using sms_df
sms_df['type_encoded'] = sms_df['type'].map({'ham': 0, 'spam': 1})

# 5. Extract features and target from sms_df
X_raw = sms_df['text']
y = sms_df['type_encoded']

# ------------------------------------------------------------------------------
# STEP 2: TRAIN-TEST SPLIT (Executed before vectorization to avoid data leakage)
# ------------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------------------
# 5.1 Build the model on scaled data (Try multiple text feature scaling options)
# ------------------------------------------------------------------------------
# Option 1: TF-IDF Scaling (Normalizes term frequencies relative to document lengths)
tfidf_vec = TfidfVectorizer(max_features=4000, min_df=2, stop_words='english')
X_train_tfidf = tfidf_vec.fit_transform(X_train_raw)
X_test_tfidf = tfidf_vec.transform(X_test_raw)

# Option 2: Binary Count Scaling (Strict presence/absence flags used by Bernoulli)
count_vec = CountVectorizer(max_features=4000, min_df=2, stop_words='english', binary=True)
X_train_count = count_vec.fit_transform(X_train_raw)
X_test_count = count_vec.transform(X_test_raw)

# Handle class imbalance on the TF-IDF feature matrix via SMOTE
smote = SMOTE(random_state=42)
X_train_res_tfidf, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# ------------------------------------------------------------------------------
# 5.2 Build a Naïve Bayes Model
# ------------------------------------------------------------------------------
# Base Model A: Multinomial Naïve Bayes on TF-IDF Scaled Data
mnb_model = MultinomialNB()
mnb_model.fit(X_train_res_tfidf, y_train_res)

# Base Model B: Bernoulli Naïve Bayes on Binary Sparsity Count Data
X_train_res_count, y_train_res_count = smote.fit_resample(X_train_count, y_train)
bnb_model = BernoulliNB()
bnb_model.fit(X_train_res_count, y_train_res_count)

# ------------------------------------------------------------------------------
# 5.3 Validate the model with test data and obtain evaluation metrics
# ------------------------------------------------------------------------------
y_pred_mnb = mnb_model.predict(X_test_tfidf)

print("--- BASELINE MULTINOMIAL NAIVE BAYES PERFORMANCE ---")
print("Test Accuracy :", accuracy_score(y_test, y_pred_mnb))
print("Precision     :", precision_score(y_test, y_pred_mnb))
print("Recall        :", recall_score(y_test, y_pred_mnb))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_mnb))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_mnb))

# ------------------------------------------------------------------------------
# 5.4 Tune the model and improve the accuracy
# ------------------------------------------------------------------------------
# Fine-tuning MultinomialNB using Laplace smoothing parameter 'alpha' to maximize F1-score
param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_res_tfidf, y_train_res)

best_sms_model = grid_search.best_estimator_
y_test_pred_tuned = best_sms_model.predict(X_test_tfidf)

print("\n--- TUNED MULTINOMIAL NAIVE BAYES PERFORMANCE ---")
print("Best Hyperparameter Settings (alpha):", grid_search.best_params_)
print("Tuned Test Accuracy :", accuracy_score(y_test, y_test_pred_tuned))
print("Tuned Precision     :", precision_score(y_test, y_test_pred_tuned))
print("Tuned Recall        :", recall_score(y_test, y_test_pred_tuned))
print("\nTuned Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred_tuned))
print("\nTuned Classification Report (Test):\n", classification_report(y_test, y_test_pred_tuned))
#===========================================================================================
6. Benefits and Business Impact of the Solution
#=============================================================================================
Implementing an automated, highly accurate SMS spam filtering tool provides significant 
financial and structural benefits to telecommunication companies, enterprise communication
 platforms, and endpoint users:

Enhanced User Safety and Protection Against Phishing: Cybercriminals use text messaging 
streams to run deceptive sweepstakes, financial phishing traps, and fraudulent account 
collections. This model flags and blocks harmful texts before they reach the inbox, shielding 
users from identity theft and financial scams.

Preservation of Mobile Network Bandwidth and Lower Infrastructure Costs: Filtering out large 
volumes of automated industrial spam text traffic reduces the bandwidth load on core network
 servers. Telecom clients can optimize hardware usage and lower processing costs by deflecting
 garbage payloads at the edge.

Higher Trust and Customer Retention for Messaging Platforms: Mobile applications or carrier 
networks that guarantee a clean, spam-free messaging environment build higher user
 satisfaction. Minimizing inbox clutter directly reduces churn rates and improves subscriber
 lifetime value.

Low Computational Overhead for Instant Carrier Filtering: Unlike deep learning network 
variations that demand expensive GPU infrastructure, Naïve Bayes relies on rapid, lightweight 
multiplication operations. This allows corporate entities to deploy the model directly inside 
high-throughput live routers to screen millions of incoming text packages per second without
 adding network latency.