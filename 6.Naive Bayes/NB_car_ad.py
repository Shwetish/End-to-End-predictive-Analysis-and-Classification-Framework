#=======================================================================
#Business Understanding:-
#=======================================================================
#Business Understanding:-
#1. Business Problem Statement
#"A car company is launching a new product and running social media 
#advertisements. Right now, they are showing ads to everyone blindly,
#which wastes a lot of marketing budget on people who have no intention
#or capacity to buy a car. They need to know exactly who is likely to 
#click and buy so they can target only the right audience."

#2. Business Objective
#Analyze historical customer walk-in and ad click data (based on age
#and salary) to build a smart classification model. The manager can 
#use this model to predict whether a specific user on social media will
#buy the car (1) or not (0), allowing them to show ads only to 
#high-potential buyers.

#3. Motivation 
#Saving Budget: Instead of showing ads to a 19-year-old making ₹19,000 
#(who is highly unlikely to buy), the system automatically shifts focus
# to a 47-year-old earning ₹25,000 or a 32-year-old earning ₹150,000 
#who are ready to buy.
#Higher Conversion Rates: Showing the right ad to the right person at 
#the right time leads to more actual car sales with fewer ad views.

#4. Constraints
#Simple Feature Profile: The company only collects basic information—
#Gender, Age, and Estimated Salary. The model must find strong patterns
#using just these few details without needing complex historical credit
#data.
#No Privacy Violations: The User ID is a random identification number
#that does not reveal personal private data, meaning it cannot be used 
#for direct ad decision-making.

#5. Business Success Criteria
#Reduce marketing ad spend waste by 20% within the first campaign.
#Increase the click-to-purchase conversion rate of the social media 
#ads.

#6. ML Success Criteria
#Train a classification model (like Naive Bayes or Logistic Regression)
#that successfully splits customers into Purchased (1) or Not Purchased
#(0) categories.
#Achieve an overall model Accuracy score of ≥ 80% on new test data 
#before launching the live ad system.

#======================================================================
#Data Understanding:-
#======================================================================
'''
Name of Feature           Description           Type          Relevance
User ID          A unique random number       Quantitative,   Irrelevant
               assigned to each individual      Nominal
                     customer.      
Gender          The biological sex of the     Qualitative
               customer (Male or Female).,     Categorical     Medium
Age             The exact age of the customer Quantitative,
                   in years.                  Continuous       High
Estimated       The calculated yearly         Quantitative,
 Salary        income of the customer.        Continuous       High
Purchased      Shows if the customer bought   Qualitative
               the car (1 = Yes, 0 = No).     Nominal          High(Target Variable)
'''
# ==============================================================================
# Exploratory Data Analysis (EDA) with Inline Inference - Car Advertisement
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# -----------------------------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------------------------
# Replace with your actual local file path
car_ad = pd.read_csv('C:/15_Naive_Byes_salary_Data_2026/NB_Car_Ad .csv')

# -----------------------------------------------------------------------------
# Basic Understanding
# -----------------------------------------------------------------------------
car_ad.head()
car_ad.info()
car_ad.describe()

# Inference:
# Dataset contains a unique identifier (User ID), demographic variables 
# (Gender, Age), a financial indicator (Estimated Salary), and the target variable.
# Target variable 'Purchased' is categorical (0 or 1) -> classification problem.

# -----------------------------------------
# Target Variable Distribution
# -----------------------------------------
car_ad['Purchased'].value_counts()
sns.countplot(x='Purchased', data=car_ad)
plt.show()

# Inference:
# Dataset is somewhat imbalanced.
# The majority of targeted users did not purchase the car after seeing the ad (Purchased = 0).
# This baseline class balance must be kept in mind during model evaluation.

#---------------------------------------------------------------
# Age Distribution
#------------------------------------------------------------
sns.displot(car_ad['Age'], kde=True)
plt.show()

# Inference:
# Age distribution is close to a broad, uniform/normal-like curve.
# Most individuals fall within the core active working-age range of 25 to 50 years old.

#----------------------------------------------------------------------
# Estimated Salary Distribution
#----------------------------------------------------------------------
sns.displot(car_ad['EstimatedSalary'], kde=True)
plt.show()
# Inference:
# Estimated salary distribution is multi-modal and slightly right-skewed.
# A large portion of users earn mid-range salaries, with smaller peaks at high income tiers.

#--------------------------------------------------------------------
# Univariate Boxplot - Age
#---------------------------------------------------------------------
sns.boxplot(x=car_ad['Age'])
plt.show()

# Inference:
# The median age sits safely around 37 years old.
# No extreme outliers are present in the age dimension; the distribution is well-bounded.

#-------------------------------------------------------------
# Univariate Boxplot - Estimated Salary
#------------------------------------------------------------
sns.boxplot(x=car_ad['EstimatedSalary'])
plt.show()

# Inference:
# Salaries are well-distributed across a wide range ($15,000 to $150,000).
# No out-of-bounds outliers exist, confirming a clean sample of consumer income profiles.

# -----------------------------------------------------------
# Correlation Among Numerical Features
# -----------------------------------------------------------
sns.heatmap(
    car_ad[['Age', 'EstimatedSalary']].corr(),
    annot=True, cmap='coolwarm'
)
plt.show()

# Inference:
# Age and Estimated Salary display a very weak positive correlation.
# This indicates that salary doesn't simply linearly scale with age in this dataset.

#----------------------------------------------------------
# Gender vs Purchased Conversion
#------------------------------------------------------------
sns.countplot(x='Gender', hue='Purchased', data=car_ad)
plt.show()

# Inference:
# Both genders show relatively similar purchase conversion ratios.
# Gender alone might carry less predictive weight compared to age or salary metrics.

#-------------------------------------------------------------
# Bivariate Analysis - Age vs Estimated Salary split by Purchased
# -------------------------------------------------------------
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=car_ad, palette='coolwarm')
plt.show()

# Inference:
# A very clear visual pattern emerges: users who are older (above 40) OR 
# have higher salaries (above $75,000) show a massive clustering of purchases.
# Younger, lower-earning profiles consistently ignore the car ad.

#-------------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS (Alternative Plots)
#-------------------------------------------------------------------------
plt.hist(car_ad.Age, bins=15, color='skyblue', edgecolor='black')
plt.title("Age Distribution Histogram")
plt.show()

plt.hist(car_ad.EstimatedSalary, bins=15, color='salmon', edgecolor='black')
plt.title("Estimated Salary Distribution Histogram")
plt.show()

#-----------------------------------------------------------------------
# Final EDA Conclusion
#------------------------------------------------------------------------
# The car ad dataset represents a clear consumer conversion classification problem.
# Purchasing decisions are heavily influenced by a combination of higher age and salary.
# No major outlier treatment is required since metrics fall within normal human boundaries.
# 'User ID' must be removed before training as it is an uninformative unique sequence key.
# 'Gender' will require clean binary encoding (0/1) for Naïve Bayes processing.

# ==============================================================================
# Data Preprocessing for Car Advertisement Dataset
# ==============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load Dataset
# Replace with your actual local file path
df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/NB_Car_Ad .csv')
print("Initial Shape:", df.shape)
df.head()

# Data Types Check
df.info()

# Inference:
# Dataset contains numerical descriptors (Age, Estimated Salary), an ID, and a categorical string (Gender).
# Target variable 'Purchased' is numeric/categorical binary (0 or 1) → classification problem.

#---------------------------------------------------------------------
# Missing Value Analysis
#---------------------------------------------------------------------
print("Missing values:\n", df.isnull().sum())

# Inference:
# No missing values detected in the dataset features or the target variable.
# The data is complete, but we keep structural treatment steps for pipeline safety.

#-----------------------------------------------------------------------
# Missing Value Treatment
#-------------------------------------------------------------------------
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Fill categorical missing values (if any were to occur dynamically)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numerical missing values with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Inference:
# Mode fallback preserves textual groups like Gender.
# Median strategy ensures safety from scaling shifts if single values are missing.

#-----------------------------------------------------------------------
# Duplicate Removal
#--------------------------------------------------------------------------
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# Inference:
# Removing duplicates prevents identical user behavior rows from biasing the model.
# This ensures that validation sets evaluate true generalized accuracy.

# ------------------------------------------------------------------------------
# Univariate Outlier Detection (Numerical Features)
# ------------------------------------------------------------------------------
# Excluding ID and Target boundaries from feature outlier checks
check_cols = ['Age', 'Estimated']
for col in check_cols:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# Inference:
# Age and Estimated Salary distributions sit cleanly within realistic operational bounds.
# No extreme outlier point spikes are visible in these core attributes.

# -------------------------------------------------------------------------
# Winsorization (Outlier Treatment)
# ----------------------------------------------------------------------------
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['Age', 'EstimatedSalary']
)

df[['Age', 'EstimatedSalary']] = winsor.fit_transform(df[['Age', 'EstimatedSalary']])
sns.boxplot(x=df.Age)
plt.show()

# Inference:
# Capping normalizes values at 1.5 IQR thresholds.
# This helps maintain safe data distribution boundaries for parametric models.

#---------------------------------------------------------------------
# Encoding Target Variable
#-------------------------------------------------------------------
# Target 'Purchased' is already in 0 and 1 format in our raw data file.
# We ensure structural integrity via casting.
df['Purchased'] = df['Purchased'].astype(int)

# Inference:
# Target format verification confirmed: 0 -> Didn't Purchase, 1 -> Purchased.
# Perfectly structured for binary probability evaluations.

#-------------------------------------------------------------------
# One-Hot Encoding Categorical Features
#---------------------------------------------------------------------
# Dropping non-predictive unique database identifier 'User ID' first
df.drop(columns=['User ID'], inplace=True, errors='ignore')

df_encoded = pd.get_dummies(df, drop_first=True)
print("Shape after encoding:", df_encoded.shape)

# Inference:
# Converts 'Gender' text into numerical indicator blocks ('Gender_Male').
# Dropping the 'User ID' sequence column keeps the system from over-fitting to row numbers.

#------------------------------------------------------------------------
# Zero Variance Feature Removal
#---------------------------------------------------------------------------
X = df_encoded.drop(columns='Purchased')
y = df_encoded['Purchased']

selector = VarianceThreshold(threshold=0.0)
X_var = selector.fit_transform(X)

# Inference:
# Checks for and removes columns where every row holds the exact same value.
# Ensures that all processed attributes contribute distinct information to the model.

#------------------------------------------------------------------------
# Feature Scaling
#----------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inference:
# Standardization centers all features around a mean of 0 and a standard deviation of 1.
# This prevents the larger magnitude of 'Estimated Salary' from distorting probability weights relative to 'Age'.

#------------------------------------------------------------------------------
# Train-Test Split
#------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Inference:
# Stratification ensures that the train and test subsets match the original data's 0:1 conversion balance.

#---------------------------------------------------------
# Class Imbalance Handling using SMOTE
#-------------------------------------------------------
print("Before SMOTE:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE:\n", pd.Series(y_train_res).value_counts())

# Inference:
# Synthetically aligns the minority class 'Purchased' count with the majority class.
# This prevents the Naïve Bayes model from developing a structural bias 
#toward guessing 0.

# ==============================================================================
# Naive Bayes Model Development: Car Advertisement Dataset (NB_Car_Ad)
# ==============================================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------------------
# STEP 1: LOAD DATASET & BASIC SEPARATION
# ------------------------------------------------------------------------------
# Replace with your actual local file path
df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/NB_Car_Ad .csv')

# Drop non-predictive User ID column
if 'User ID' in df.columns:
    df.drop(columns=['User ID'], inplace=True)

X = df.drop(columns='Purchased')
y = df['Purchased']

# ------------------------------------------------------------------------------
# STEP 2: SPLIT DATASET (Train-Test Split)
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------------------
# STEP 3: PREPROCESSING & ENCODING (Strictly fit on train, transform on test)
# ------------------------------------------------------------------------------
# Encode Categorical Feature 'Gender'
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align columns just in case
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ------------------------------------------------------------------------------
# 5.1 Build the model on the scaled data (Try multiple options) & 5.2 Naive Bayes
# ------------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balance training data classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Option A: Baseline Gaussian Naive Bayes Model
gnb_model = GaussianNB()
gnb_model.fit(X_train_res, y_train_res)

# Option B: Bernoulli Naive Bayes Model (Alternative option to test)
bnb_model = BernoulliNB()
bnb_model.fit(X_train_res, y_train_res)

# ------------------------------------------------------------------------------
# 5.3 Validate the model with test data (Confusion Matrix, Precision, Recall, Accuracy)
# ------------------------------------------------------------------------------
y_pred_gnb = gnb_model.predict(X_test_scaled)

print("--- BASELINE GAUSSIAN NAIVE BAYES PERFORMANCE ---")
print("Test Accuracy :", accuracy_score(y_test, y_pred_gnb))
print("Precision     :", precision_score(y_test, y_pred_gnb))
print("Recall        :", recall_score(y_test, y_pred_gnb))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_gnb))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_gnb))

# ------------------------------------------------------------------------------
# 5.4 Tune the model and improve the accuracy
# ------------------------------------------------------------------------------
# Tuning GaussianNB using Var_smoothing to adjust variance bounds and counter skewness
param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
grid_search = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

best_nb_model = grid_search.best_estimator_
y_test_pred_tuned = best_nb_model.predict(X_test_scaled)
y_train_pred_tuned = best_nb_model.predict(X_train_scaled)

print("\n--- TUNED GAUSSIAN NAIVE BAYES PERFORMANCE ---")
print("Best Parameters:", grid_search.best_params_)
print("Tuned Test Accuracy :", accuracy_score(y_test, y_test_pred_tuned))
print("Tuned Precision     :", precision_score(y_test, y_test_pred_tuned))
print("Tuned Recall        :", recall_score(y_test, y_test_pred_tuned))
print("\nTuned Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred_tuned))
#===========================================================================
#Benefits and Business Impact of the Solution:-
#===========================================================================
#Implementing this targeted Naïve Bayes predictive classification framework
# provides measurable, bottom-line benefits for the dealership or car brand's
# social media marketing operations:

#Drastic Reduction in Ad Spend Waste (Optimized ROI): Instead of broadcasting
#expensive social media ad impressions broadly across all age brackets and 
#salary bounds, the sales team can run pre-filtered lookalike audiences. 
#By ignoring low-probability segments (e.g., younger users with entry-level
# estimated wages), the dealership ensures that every dollar of budget
# targets high-probability buyers.
#Automated Lead Prioritization for Sales Pipelines: The Naïve Bayes output 
#calculates real-time conversion probabilities. Sales representatives can 
#use these probabilities to prioritize high-value prospects (such as those
# matching high-income brackets) for personalized outreach or premium 
#test-drive bookings.
#Tailored Ad Creatives and Campaign Personalization: Knowing that conversions
# strongly cluster around distinct demographic groups allows creative teams 
#to design targeted ads. High-income segments can be served luxury vehicle 
#campaigns, while other profiles see cost-effective or family-oriented ad 
#creatives.
#Lightweight and Fast Real-Time Inferences: Because Naïve Bayes relies on 
#simple conditional probability math, the trained model takes up minimal 
#memory and runs instantly. This allows it to be embedded directly into live
# ad servers or CRM systems to score incoming customer leads on the fly 
#without system lag.