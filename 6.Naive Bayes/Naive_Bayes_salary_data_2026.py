# =============================================================================
# SALARY PREDICTION USING NAÏVE BAYES CLASSIFIER
# =============================================================================

# -----------------------------------------------------------------------------
# BUSINESS UNDERSTANDING
# -----------------------------------------------------------------------------
# 1. Business Problem Statement:
# - Organizations want to predict whether an employee earns
# - more than 50K or less/equal to 50K per year based on
# - Demographic and job-related attributes.

# 2. Business Objective:
# - Predict salary category accurately
# - Support HR decisions such as compensation planning
# - Enable workforce analytics

# 3. Motivation:
#   Understanding employee salary patterns helps businesses:
# - Retain talent
# - Design fair compensation str

# 4. Constraints:
# - High dimensional categorical data
# - Class imbalance problem
# - Requires large datasets for better generalization
#
# 5. Success Criteria:
# Business Success:
# - Accurate salary prediction
# ML Success:
# - Good accuracy on unseen test data
# - Balanced misclassification errors
#
# -----------------------------------------------------------------------------
# DATA UNDERSTANDING
# -----------------------------------------------------------------------------
'''
Feature Name          | Description                    |       Type
Age                   | Age of employee                |    Numeric
Workclass             | Employment type                |    Categorical
Education             | Education level                |    Categorical
EducationNo           | Years of education             |    Numeric
MaritalStatus         | Marital status                 |    Categorical
Occupation            | Job role                       |    Categorical
Relationship          | Family relationship            |    Categorical
Race                  | Race category                  |    Categorical
Sex                   | Gender                         |    Categorical
CapitalGain           | Capital gain                   |    Numeric
CapitalLoss           | Capital loss                   |    Numeric
HoursPerWeek          | Working hours per week         |    Numeric
Native                | Country of origin              |    Categorical
Salary                | Income category (>50K/<50K)    |    Target
'''
# -----------------------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------------------------------------------------------
# STEP 2: LOAD TRAIN AND TEST DATASETS
# -----------------------------------------------------------------------------
salary_train = pd.read_csv("c:/15_Naive_Byes_salary_data_2026/SalaryData_Train.csv")
salary_test  = pd.read_csv("c:/15_Naive_Byes_salary_data_2026/SalaryData_Test.csv")

print(salary_train.head())

# ---------------------------------------------------------
# STEP 3: BASIC EDA
# ---------------------------------------------------------
print("\nData Types:\n", salary_train.dtypes)
print("\nSummary Statistics:\n", salary_train.describe())

# Missing values check
print("\nMissing Values (Train):\n", salary_train.isna().sum())
print("\nMissing Values (Test):\n", salary_test.isna().sum())

# ---------------------------------------------------------
# BUSINESS MOMENT DECISIONS
# ---------------------------------------------------------

# -----------------------------------------
# First Moment - Mean
# -----------------------------------------
print("\nMean:\n", salary_train.mean(numeric_only=True))

# Inference:
# Variance measures how widely salary values are spread around the mean.
# Higher variance indicates large salary differences among individuals.

#----------------------------------------------------------------------------
# Second Moment - Standard Deviation
#----------------------------------------------------------------------------
print("\nStandard Deviation:\n", salary_train.std(numeric_only=True))

# Inference:
# Standard deviation shows the average deviation from the mean salary.
# Larger values imply greater salary dispersion and compensation inequality.

#----------------------------------------------------------------------------
# Third Moment - Skewness
#-----------------------------------------------------------------------------
print("\nSkewness:\n", salary_train.skew(numeric_only=True))

# Inference:
# Positive skewness indicates that most salaries are clustered at lower levels
# with a few high-salary outliers, resulting in a right-skewed distribution.

#-----------------------------------------------------------------------------
# Fourth Moment - Kurtosis
#----------------------------------------------------------------------------
print("\nKurtosis:\n", salary_train.kurtosis(numeric_only=True))

# Inference:
# High kurtosis suggests the presence of extreme salary values (outliers).
# The distribution is heavy-tailed and may require outlier treatment.


# Overall EDA Conclusion
#The salary dataset exhibits variability, right skewness, and potential outliers.
#Appropriate preprocessing such as scaling, transformation, or outlier handling
#is recommended before applying machine learning models.

#===============================================================================
# Exploratory Data Analysis (EDA) with Inline Inference
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# ----------------------------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------------------------
salary = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/SalaryData_Train.csv')

# -----------------------------------------------------------------------------
# Basic Understanding
# -----------------------------------------------------------------------------
salary.head()
salary.info()
salary.describe()

# Inference:
# Dataset contains both numerical and categorical variables.
# Target variable 'Salary' is categorical, indicating a
# classification problem (<=50K and >50K).

# -----------------------------------------
# Target Variable Distribution
# -----------------------------------------
salary['Salary'].value_counts()
sns.countplot(x='Salary', data=salary)
plt.show()

# Inference:
# Dataset is imbalanced.
# Majority of individuals earn <=50K.
# This imbalance must be considered during model evaluation.
#---------------------------------------------------------------
# Age Distribution
#--------------------------------------------------------------------
sns.displot(salary['age'], kde=True)
plt.show()

# Inference:
# Age distribution is right-skewed.
# Most individuals fall between 25 and 50 years of age.
#----------------------------------------------------------------------
# Working Hours per Week
#----------------------------------------------------------------------
sns.displot(salary['hoursperweek'], kde=True)
plt.show()

# Inference:
# Most people work around 40 hours per week.
# Few individuals work significantly more, creating right skewness.

# Capital Gain
sns.displot(salary['capitalgain'], kde=True)
plt.show()

# Inference:
# Capital gain is highly right-skewed.
# Majority of individuals have zero capital gain.
# Few individuals have very high investment income.

# Capital Loss
sns.displot(salary['capitalloss'], kde=True)
plt.show()

# Inference:
# Capital loss values are mostly zero.
# Only a small subset of individual

#--------------------------------------------------------------------
# Univariate Boxplot - Age
#---------------------------------------------------------------------
sns.boxplot(x=salary['age'])
plt.show()

# Inference:
# The median age lies around the middle of the working-age range.
# The distribution is slightly right-skewed.
# Few high-age outliers exist, representing older individuals still working

#-------------------------------------------------------------
# Univariate Boxplot - Education
#------------------------------------------------------------

sns.boxplot(x=salary['educationno'])
plt.show()

# Inference:
# Education levels are concentrated around mid-range values.
# The median education level indicates most individuals have
# moderate educational qualifications.
# Very low and very high education levels appear as outliers.

# ---------------------------------------------------
# Univariate Boxplot - Hours per Week
# ------------------------------------------------------------
sns.boxplot(x=salary['hoursperweek'])
plt.show()

# Inference:
# Median working hours are around 40 hours per week.
# Right-side outliers indicate individuals working unusually long hours.
# Most observations are tightly clustered around standard work hours.

#-------------------------------------------------------------------
# Univariate Boxplot - Capital Gain
#--------------------------------------------------------------------

sns.boxplot(x=salary['capitalgain'])
plt.show()

# Inference:
# Capital gain distribution is extremely right-s
# Majority of individuals have zero capital gain
# Few extreme outliers represent high investment
#------------------------------------------------------------------------------
# Univariate Boxplot - Capital Loss
#------------------------------------------------------------------------------ 
sns.boxplot(x=salary['capitalloss'])
plt.show()


# Inference:
# Capital loss values are mostly zero.
# Very few individuals experience significant capital loss.
# Presence of outliers indicates rare but large financial losses.



# Correlation Among Numerical Features
sns.heatmap(
    salary[['age','educationno','capitalgain',
    'capitalloss','hoursperweek']].corr(),
    annot=True, cmap='coolwarm'
)
plt.show()


# Inference:
# Education number and age show positive relationships
# with income-related attributes.
# Capital gain/loss show weak correlation due to sparsity.
#----------------------------------------------------------
# Gender vs Salary
#------------------------------------------------------------
sns.countplot(x='sex', hue='Salary', data=salary)
plt.show()

# Inference:
# Males have a higher proportion of >50k earners.
# Females are more concentrated in the <=50K category.
#-------------------------------------------------------------
# Workclass Distribution
# -------------------------------------------------------------
sns.countplot(y='workclass', data=salary)
plt.show()

# Inference:
# Majority of individuals belong to the Private workclass.
# Government and self-employed categories are less frequent.
#--------------------------------------------------------------
# Occupation Distribution
#---------------------------------------------------------------
sns.countplot(y='occupation', data=salary)
plt.show()

# Inference:
# Professional and white-collar occupations are more common.
# Income levels vary significantly across occupations.

#-----------------------------------------------------------------------
# Final EDA Conclusion
#------------------------------------------------------------------------
# The dataset represents a real-world income classification problem.
# Salary is influenced by age, education level, working hours, and gender.
# Capital gain/loss contain extreme values and sparsity.
# Categorical variables require encoding before model training.
# Class imbalance should be handled during machine learning.

#-------------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS
#-------------------------------------------------------------------------
plt.hist(salary_train.age)
plt.title("Age Distribution")
plt.show()

plt.hist(salary_train.educationno)
plt.title("Education Number Distribution")
plt.show()

#-------------------------------------------------------------------------
# Data Preprocessing for Salary Classification Dataset
#-------------------------------------------------------------------------
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
df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/SalaryData_Train.csv')
print("Initial Shape:", df.shape)
df.head()

# Data Types Check
df.info()

# Inference:
# Dataset contains both numerical and categorical variables.
# Target variable 'Salary' is categorical → classification problem.

#---------------------------------------------------------------------
# Missing Value Analysis
#---------------------------------------------------------------------
print("Missing values:\n", df.isnull().sum())

#-----------------------------------------------------------------------
# Missing Value Treatment
#-------------------------------------------------------------------------
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Fill categorical missing values 
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numerical missing values with median 
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Inference:
# Mode is suitable for categorical features.
# Median is robust to outliers in numerical features.
#-----------------------------------------------------------------------
# Duplicate Removal
#--------------------------------------------------------------------------
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# Inference:
# Duplicate records may bias learning algorithms.
# Removing them improves model generalization.

# ------------------------------------------------------------------------------
# Univariate Outlier Detection (Numerical Features)
# ------------------------------------------------------------------------------

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

# Inference:
# Capital gain and capital loss show extreme outliers.
# Age and hourperweek show mild skewness.

# -------------------------------------------------------------------------
# Winsorization (Outlier Treatment)
# ----------------------------------------------------------------------------
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['age', 'hoursperweek', 'educationno']
)

df[['age','hoursperweek','educationno']] = winsor.fit_transform(
    df[['age','hoursperweek','educationno']]
)
sns.boxplot(df.age)

# Inference:
#winsorization caps extreme values without deleting data.
#Helps stabilize model coefficients.
#---------------------------------------------------------------------
# Encoding Target Variable
#-------------------------------------------------------------------
le = LabelEncoder()
df['Salary'] = le.fit_transform(df['Salary'])

# Inference:
# <=50K -> 0, >50K -> 1
# Required for machine learning algorithms.
#-------------------------------------------------------------------
# One-Hot Encoding Categorical Features
#---------------------------------------------------------------------
df_encoded = pd.get_dummies(df, drop_first=True)
print("Shape after encoding:", df_encoded.shape)

# Inference:
# Converts categorical variables into numeric format.
# drop_first=True avoids dummy variable trap.
#------------------------------------------------------------------------
# Zero Variance Feature Removal
#---------------------------------------------------------------------------
selector = VarianceThreshold(threshold=0.0)
df_var = selector.fit_transform(df_encoded)

# Inference:
# Removes constant features that add no predictive value.
#------------------------------------------------------------------------
# Feature Scaling
#----------------------------------------------------------------------------
X = df_encoded.drop(columns='Salary')
y = df_encoded['Salary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inference:
# Standardization ensures all features are on the same scale.
# Important for distance-based

#------------------------------------------------------------------------------
# Train-Test Split
#------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Inference:
# Stratified split maintains class distribution.
#---------------------------------------------------------
# Class Imbalance Handling using SMOTE
#-------------------------------------------------------
#pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0

print("Before SMOTE:\n", y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE:\n", pd.Series(y_train_res).value_counts())
#==============================================================================
# Naive Bayes Model Development (Two Separate Datasets)
#==============================================================================
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
#==============================================================================
# STEP 1: LOAD TRAIN AND TEST DATASETS
#==============================================================================
train_df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/SalaryData_Train.csv')
test_df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/SalaryData_Test.csv')

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)
#-------------------------------------------
#step2:Separate features and the Target
# -----------------------------------------------
X_train = train_df.drop(columns='Salary')
y_train = train_df['Salary']

X_test = test_df.drop(columns='Salary')
y_test = test_df['Salary']

# -------------------------------------------------------------
# STEP 3: HANDLE MISSING VALUES (TRAIN & TEST)
# ------------------------------------------------------------------

cat_cols = X_train.select_dtypes(include='object').columns
num_cols = X_train.select_dtypes(include=['int64','float64']).columns

for col in cat_cols:
    X_train[col].fillna(X_train[col].mode()[0])
    X_test[col].fillna(X_train[col].mode()[0])

for col in num_cols:
    X_train[col].fillna(X_train[col].median())
    X_test[col].fillna(X_train[col].median())
 #------------------------------------------------------------   
# STEP 4: ENCODE TARGET VARIABLE
#--------------------------------------------------------------
le = LabelEncoder()
y_train = le.fit_transform(y_train)    # fit only on train
y_test = le.transform(y_test)

# <=50K → 0, >50K → 1
#-------------------------------------------------------------
# STEP 5: ONE-HOT ENCODE CATEGORICAL FEATURES
#-------------------------------------------------------------
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align test columns with train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
#---------------------------------------------------------------
# STEP 6: FEATURE SCALING    
#-----------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#-----------------------------------------------------------------
# STEP 7: HANDLE CLASS IMBALANCE (TRAIN ONLY)
#-----------------------------------------------------------------
print("Before SMOTE:\n", pd.Series(y_train).value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("After SMOTE:\n", pd.Series(y_train_res).value_counts())

#---------------------------------------------------------------------
# STEP 8: GAUSSIAN NAIVE BAYES MODEL
#--------------------------------------------------------------------
nb_model = GaussianNB()
nb_model.fit(X_train_res, y_train_res)

# --------------------------------------------------------------------
# STEP 9: PREDICTION
# --------------------------------------------------------------------
y_test_pred = nb_model.predict(X_test_scaled)
y_train_pred = nb_model.predict(X_train_scaled)

# ---------------------------------------------------------------------
# STEP 10: MODEL EVALUATION - TEST DATA
# ------------------------------------------------------------------------
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

#-----------------------------------------------------------------------
# STEP 11: MODEL EVALUATION - TRAIN DATA
#------------------------------------------------------------------------
print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nConfusion Matrix (Train):\n", confusion_matrix(y_train, y_train_pred))

#-------------------------------------------------------------------------
# FINAL BUSINESS CONCLUSION
#---------------------------------------------------------------------------
'''
- **Binary salary classification problem** (<=50K vs >50K)
- **Gaussian Naive Bayes suits continuous, scaled features**
- **SMOTE** improves minority class (>50K) learning
- **Test performance reflects real-world generalization**
- **Model acts as a fast, interpretable baseline for HR analytics**

'''

# Why results are poor  
Naive Bayes Assumption is Violated  
Gaussian Naive Bayes assumes:  
Features are independent  
Features follow a normal distribution
In the Salary dataset:
Age, education, hoursperweek, capitalgain are strongly dependent
Distributions are highly skewed, not Gaussian
Result: Model cannot learn correct class boundaries.
Heavy Class Overlap
Even after SMOTE:
Many <=50K and >50K samples overlap
Same age / education / hours → different salary classes  
Result: Probabilistic separation becomes weak.  

GaussianNB + One-Hot Encoding  

One-hot encoding creates binary sparse features  

GaussianNB is not ideal for binary dummy variables

SMOTE + GaussianNB Interaction

SMOTE generates synthetic continuous values

GaussianNB treats them as real distributions

Result: Probability density estimation becomes noisy.

Accuracy is a weak Metric Here

With class imbalance:

Accuracy = 50-60% can happen even with poor discrimination

Model may be predicting both classes almost randomly

Better metrics: Recall, F1-score, ROC-AUC

'''




































































































































# Inference:
# Education number and age show positive relationships
# with income-related attributes.
# Capital gain/loss show weak correlation due to sparsity.
#----------------------------------------------------------
# Gender vs Salary
#------------------------------------------------------------
sns.countplot(x='sex', hue='Salary', data=salary)
plt.show()

# Inference:
# Males have a higher proportion of >50k earners.
# Females are more concentrated in the <=50K category.
#-------------------------------------------------------------
# Workclass Distribution
# -------------------------------------------------------------
sns.countplot(y='workclass', data=salary)
plt.show()

# Inference:
# Majority of individuals belong to the Private workclass.
# Government and self-employed categories are less frequent.
#--------------------------------------------------------------
# Occupation Distribution
#---------------------------------------------------------------
sns.countplot(y='occupation', data=salary)
plt.show()

# Inference:
# Professional and white-collar occupations are more common.
# Income levels vary significantly across occupations.

#-----------------------------------------------------------------------
# Final EDA Conclusion
#------------------------------------------------------------------------
# The dataset represents a real-world income classification problem.
# Salary is influenced by age, education level, working hours, and gender.
# Capital gain/loss contain extreme values and sparsity.
# Categorical variables require encoding before model training.
# Class imbalance should be handled during machine learning.

#-------------------------------------------------------------------------
# Data Preprocessing for Salary Classification Dataset
#-------------------------------------------------------------------------
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
df = pd.read_csv('c:/15_Naive_Byes_salary_data_2026/SalaryData_Train.csv')
print("Initial Shape:", df.shape)
df.head()
# ---------------------------------------------------------
# DATA VISUALIZATION - SUMMARY INFERENCES
# ---------------------------------------------------------

# Inference on Dataset Balance:
# The dataset is imbalanced; the majority of individuals earn <=50K.
# This imbalance must be carefully considered during model evaluation to 
# avoid biased predictions.

# Inference on Outliers:
# Boxplots for 'age', 'educationno', and 'hoursperweek' reveal the presence 
# of outliers.
# Very low and very high education levels appear as outliers relative to the 
# median educational qualification.

# Inference:
# Dataset is imbalanced.
# Majority of individuals earn <=50K.
# This imbalance must be considered during model evaluation.

# ---------------------------------------------------------
# Age Distribution
# ---------------------------------------------------------
sns.displot(salary['age'], kde=True)
plt.show()

# Inference:
# Age distribution is right-skewed.
# Most individuals fall between 25 and 50 years of age.

# ---------------------------------------------------------
# Working Hours per Week
# ---------------------------------------------------------
sns.displot(salary['hoursperweek'], kde=True)
plt.show()



# Capital gain is highly right-skewed.
# Majority of individuals have zero capital gain.
# Few individuals have very high investment income.
#
# -----------------------------------------------------------------
# Capital Loss
# -----------------------------------------------------------------
sns.displot(salary['capitalloss'], kde=True)
plt.show()

# Inference:
# Capital loss values are mostly zero.
# Only a small subset of individuals report losses.

# -----------------------------------------------------------------
# Univariate Boxplot - Age
# -----------------------------------------------------------------
sns.boxplot(x=salary['age'])
plt.show()

# Inference:
    
# -----------------------------------------------------------------
# Univariate Boxplot - Education Number
# -----------------------------------------------------------------
sns.boxplot(x=salary['educationno'])
plt.show()

# Inference:
# Education levels are concentrated around mid-range values.
# The median education level indicates most individuals have
# moderate educational qualifications.
# Very low and very high education levels appear as outliers.

# -----------------------------------------------------------------
# Univariate Boxplot - Hours per Week
# -----------------------------------------------------------------
sns.boxplot(x=salary['hoursperweek'])
plt.show()

# Inference:
# Median working hours are aroun











