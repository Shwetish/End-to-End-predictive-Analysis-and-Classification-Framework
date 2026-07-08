#=============================================================================
# Business Understanding:-
#=============================================================================
1️.Business Problem Statement
When workers leave a company too fast, it hurts the business. Managers need to know exactly how much a low salary raise drives employees to quit.

2.️Business Objective
Find the Link: See exactly how much giving bigger salary raises helps stop employees 
from quitting.

Build a Predictor: Create a smart tool that guesses what the quitting rate will be 
based on the size of the raise.

Smarter Planning: Use this data to plan better budgets and design fairer pay plans to
 keep workers happy.

3️.Motivation
It stops top talent from leaving the company.

It helps design good pay raises without wasting company money.

It warns leadership ahead of time if a specific department is at risk of losing too
 many people.

4️.Constraints
The Human Element: Every employee has different reasons for staying or leaving that
 numbers cannot easily measure.

Small Data: There might be only a tiny amount of past company data to study.

Curved Patterns: The link between money and quitting might be a curve rather than a
 perfectly straight line.

Boss-Level Raises: A few rare, massive executive pay raises can mess up regular math 
formulas.

5️.Success Criteria
Business Success: Management can accurately guess quitting rates so leadership can
 take action to keep the best workers.

ML Success: The AI model makes tiny guessing errors (low RMSE), proves a strong link 
where higher pay means fewer people quit, and works great on new data.
-----------------------------------------------------------------------
DATA UNDERSTANDING:-
-----------------------------------------------------------------------
'''
Feature Name        Description Type              Business Relevance
Salary_hike         Salary hike amount            Numeric Key predictor
Churn_out_rate      Employee churn out rate (%)   Numeric Target variable
'''
# =======================================================================
# EXPLORATORY DATA ANALYSIS 
# =======================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------------------
# STEP 2: LOAD THE DATASET
#-----------------------------------------------------------------------
# Loading the SAT-GPA academic dataset
df_acad = pd.read_csv("C:/21_Linear_Regression_Colory/SAT_GPA.csv")

# Standardizing column names based on your spreadsheet images
df_acad.columns = ["sat_scores", "gpa"]

print("First 5 Rows:\n", df_acad.head())

#-----------------------------------------------------------------------
# STEP 3: BASIC EDA
#-----------------------------------------------------------------------
print("\nData Types:\n", df_acad.dtypes)
print("\nSummary Statistics:\n", df_acad.describe())

# -----------------------------------------------------------------------
# BUSINESS MOMENT DECISIONS
# -----------------------------------------------------------------------

# Mean
print("\nMean:\n", df_acad.mean())
'''
Inference:
• Average SAT score shows the baseline standardized test performance of the student group.
• Average GPA represents the typical grade point average achieved by these students.
'''

# Variance
print("\nVariance:\n", df_acad.var())
'''
Inference:
• High variance in SAT scores -> The dataset includes a very diverse mix of students with widely varying test performances.
• High variance in GPA -> Students' actual classroom grades vary significantly across the board.
'''

# Standard Deviation
print("\nStandard Deviation:\n", df_acad.std())
'''
Inference:
• Shows the typical deviation or "gap" from the average student benchmarks.
• Higher values mean academic performance ranges from very low to very high rather than clustering tightly around the average.
'''

# Skewness
print("\nSkewness:\n", df_acad.skew())
'''
Inference:
• Skew ≈ 0 -> Symmetrical, balanced academic distribution across the student sample.
• Negative skew -> The dataset is leaning toward higher-performing individuals, with only a few struggling students pulling the tail left.
'''

# Kurtosis
print("\nKurtosis:\n", df_acad.kurtosis())
'''
Inference:
• High kurtosis -> Points to extreme clusters (e.g., unexpected spikes of perfect SAT scores or failing GPAs).
• Low kurtosis -> Uniform distribution across different grading tiers.
'''

# Correlation
print("\nCorrelation Matrix:\n", np.corrcoef(df_acad.sat_scores, df_acad.gpa))
'''
Inference:
• Expected to show a positive correlation value.
• This means a direct link: as a student's SAT score goes UP, their college GPA tends to go UP as well.
'''

# -----------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS
# -----------------------------------------------------------------------

# Histogram - SAT Scores
plt.figure(figsize=(6,4))
plt.hist(df_acad.sat_scores, color='indigo', edgecolor='black')
plt.title("SAT Scores Distribution")
plt.xlabel("SAT Score")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
• Shows where the majority of students score on their entrance exams.
• Helps spot if the exam scores are evenly distributed or heavily skewed toward top-tier/lower-tier marks.
'''

# Histogram - GPA
plt.figure(figsize=(6,4))
plt.hist(df_acad.gpa, color='forestgreen', edgecolor='black')
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
• Shows the most common grade point averages across the student population.
• High density in the 3.0–4.0 range reveals a group with strong academic achievements.
'''

#-----------------------------------------------------------------------
# STEP 4 (Continued): BOX PLOT (OUTLIER DETECTION)
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df_acad[['sat_scores', 'gpa']], orient='h', palette='Set3')
plt.title("Boxplot of SAT Scores & GPA")
plt.show()

'''
Inference:
• Outlier Detection: Helps identify academic anomalies (e.g., a student with a near-perfect SAT score who is failing their classes, or vice-versa).
• Individual points outside the whiskers represent exceptions to typical student paths.
'''

#-----------------------------------------------------------------------
# STEP 5: BIVARIATE ANALYSIS (SCATTER PLOT)
#-----------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='sat_scores', y='gpa', data=df_acad, color='darkorange')
plt.title("SAT Scores vs GPA")
plt.xlabel("SAT Scores")
plt.ylabel("GPA")
plt.show()

'''
Inference:
• UPWARD trend expected: As test scores rise, student GPAs show a clear upward trend.
• If data points form a clear diagonal slope, it verifies that entrance exams are a highly accurate indicator of future classroom success.
'''

#-----------------------------------------------------------------------
# STEP 6: CORRELATION HEATMAP
#-----------------------------------------------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(df_acad.corr(), annot=True, cmap='mako')
plt.title("Academic Performance Correlation Heatmap")
plt.show()

'''
Inference:
• A strong positive correlation number proves that SAT metrics are mathematically reliable tools for admissions planning.
• Validates that Simple Linear Regression is an appropriate tool for predicting an incoming student's potential GPA based on their test scores.
'''

#-----------------------------------------------------------------------
# PDF & CDF ANALYSIS
#-----------------------------------------------------------------------
for col in ['sat_scores', 'gpa']:
    plt.figure(figsize=(12,4))
    
    # PDF
    plt.subplot(1,2,1)
    sns.kdeplot(df_acad[col], fill=True, color='royalblue')
    plt.title(f'PDF of {col}')
    
    # CDF
    plt.subplot(1,2,2)
    sorted_vals = np.sort(df_acad[col])
    y = np.arange(len(sorted_vals)) / len(sorted_vals)
    plt.plot(sorted_vals, y, color='crimson', linewidth=2)
    plt.title(f'CDF of {col}')
    
    plt.show()
    
'''
1. PDF Analysis:
• Pinpoints the exact "sweet spot" tier where the largest segment of the student body clusters.

2. CDF Analysis:
• Helps answer academic benchmarks easily (e.g., "What percentage of our student population scored below a 3.0 GPA?").
'''

# -----------------------------------------------------------------------
# FINAL EDA SUMMARY
# -----------------------------------------------------------------------
"""
FINAL SUMMARY:-
• Confirms whether baseline test capabilities directly dictate long-term GPA benchmarks.
• Quantifies academic predictability, helping universities track and support high-risk student brackets early on.
• Linear trends justify building a predictive model to support data-driven college placement decisions.
"""
#==================================================
#Data Preprocessing
#==================================================
#-----------------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
#-----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import skew
from feature_engine.outliers import Winsorizer
import statsmodels.formula.api as smf

#-----------------------------------------------------------------------
# STEP 2: LOAD DATASET
#-----------------------------------------------------------------------
# Dependent Variable (Y): GPA, Independent Variable (X): SAT_Scores
sat_gpa = pd.read_csv("C:/21_Linear_Regression_Colory/SAT_GPA.csv")
sat_gpa.columns = ["SAT_Scores", "GPA"]

print("Initial Shape:", sat_gpa.shape)
print(sat_gpa.head())

#----------------------------------------------------------------------
# STEP 3: BASIC CLEANING
#----------------------------------------------------------------------
print("\nMissing Values Before Treatment:\n", sat_gpa.isnull().sum())

'''
Inference:
Dataset contains only numerical variables.
No identifier column present.
Missing values will be handled using robust median imputation.
'''

#-----------------------------------------------------------------------
# STEP 4: MISSING VALUE TREATMENT 
#-----------------------------------------------------------------------
for col in sat_gpa.columns:
    sat_gpa[col] = sat_gpa[col].fillna(sat_gpa[col].median())

print("\nMissing Values After Treatment:\n", sat_gpa.isnull().sum())

#-----------------------------------------------------------------------
# STEP 5: DUPLICATE REMOVAL 
#-----------------------------------------------------------------------
sat_gpa.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:", sat_gpa.shape)

#-----------------------------------------------------------------------
# STEP 6: OUTLIER DETECTION
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=sat_gpa, orient='h', palette='Set3')
plt.title("Boxplot Before Treatment")
plt.show()

#-----------------------------------------------------------------------
# STEP 7: OUTLIER TREATMENT (WINSORIZATION)
#-----------------------------------------------------------------------
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['SAT_Scores', 'GPA']
)

sat_gpa[['SAT_Scores', 'GPA']] = winsor.fit_transform(sat_gpa[['SAT_Scores', 'GPA']])

plt.figure(figsize=(8,5))
sns.boxplot(data=sat_gpa, orient='h', palette='Set3')
plt.title("Boxplot After Winsorization")
plt.show()

#-----------------------------------------------------------------------
# STEP 8: SKEWNESS CHECK
#-----------------------------------------------------------------------
print("\nSkewness:\n", sat_gpa.skew())

#-----------------------------------------------------------------------
# STEP 9: FINAL PREPROCESSING SUMMARY
#-----------------------------------------------------------------------
"""
FINAL DATA PREPROCESSING SUMMARY:
• Dataset validated and cleaned.
• Missing values handled using column-wise median.
• Duplicate records removed to maintain learning independence.
• Bounds treated using IQR-based Winsorization.
"""

#------------------------------------------------------------------------
# MODEL DEVELOPMENT - MULTIPLE LINEAR REGRESSION EXPERIMENTS
# -----------------------------------------------------------------------

# =======================================================================
# 1 SIMPLE LINEAR REGRESSION (SLR)
# =======================================================================
model1 = smf.ols('GPA ~ SAT_Scores', data=sat_gpa).fit()
pred1 = model1.predict(sat_gpa)
rmse1 = np.sqrt(np.mean((sat_gpa.GPA - pred1)**2))

print("\n--- MODEL 1: SLR ---")
print("SLR RMSE:", rmse1)
print(model1.summary())

# =======================================================================
# 2 LOG MODEL (log X)
# =======================================================================
model2 = smf.ols('GPA ~ np.log(SAT_Scores)', data=sat_gpa).fit()
pred2 = model2.predict(sat_gpa)
rmse2 = np.sqrt(np.mean((sat_gpa.GPA - pred2)**2))

print("\n--- MODEL 2: LOG-X ---")
print("Log-X Model RMSE:", rmse2)
print(model2.summary())

# =======================================================================
# 3 EXPONENTIAL MODEL (log Y)
# =======================================================================
model3 = smf.ols('np.log(GPA) ~ SAT_Scores', data=sat_gpa).fit()
pred3 = model3.predict(sat_gpa)
pred3_back = np.exp(pred3)
rmse3 = np.sqrt(np.mean((sat_gpa.GPA - pred3_back)**2))

print("\n--- MODEL 3: EXPONENTIAL ---")
print("Exponential Model RMSE:", rmse3)
print(model3.summary())

# =======================================================================
# 4 POLYNOMIAL MODEL
# =======================================================================
model4 = smf.ols('GPA ~ SAT_Scores + I(SAT_Scores**2)', data=sat_gpa).fit()
pred4 = model4.predict(sat_gpa)
rmse4 = np.sqrt(np.mean((sat_gpa.GPA - pred4)**2))

print("\n--- MODEL 4: POLYNOMIAL ---")
print("Polynomial Model RMSE:", rmse4)
print(model4.summary())

# =======================================================================
# 5. MODEL COMPARISON
# =======================================================================
results = pd.DataFrame({
    "Model": ["SLR", "Log-X", "Exponential", "Polynomial"],
    "RMSE": [rmse1, rmse2, rmse3, rmse4],
    "R_squared": [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared]
})
print("\nModel Comparison Table:\n", results)

# =======================================================================
# 6. SELECT BEST MODEL
# =======================================================================
best_model_row = results.sort_values("RMSE").iloc[0]
best_model_name = best_model_row["Model"]
print(f"\nBest Model Configuration Based on RMSE: {best_model_name}")

# =======================================================================
# 7. VISUALIZE FINAL MODEL FITS
# =======================================================================
plt.figure(figsize=(10, 6))
plt.scatter(sat_gpa.SAT_Scores, sat_gpa.GPA, color='indigo', label='Actual Student Profiles', alpha=0.6)
plt.plot(sat_gpa.SAT_Scores, pred1, color='orange', linewidth=2, label=f'SLR Fit (R²={model1.rsquared:.3f})')
plt.title("Academic Performance Regression: SAT Scores vs GPA")
plt.xlabel("SAT Scores")
plt.ylabel("GPA")
plt.legend()
plt.show()

# =======================================================================
# 9. TRAIN-TEST VALIDATION
# =======================================================================
train, test = train_test_split(sat_gpa, test_size=0.3, random_state=42)

# Fit using the designated best model approach
final_model = smf.ols('GPA ~ SAT_Scores', data=train).fit()

train_pred = final_model.predict(train)
test_pred = final_model.predict(test)

train_rmse = np.sqrt(np.mean((train.GPA - train_pred)**2))
test_rmse = np.sqrt(np.mean((test.GPA - test_pred)**2))

print("\n--- PERFORMANCE VALIDATION ---")
print("Train RMSE:", train_rmse)
print("Test RMSE :", test_rmse)
#===============================================================================
#10. Business Impact of the Project
#================================================================================
1️⃣ Data-Driven Admissions & Enrollment Filtering
Universities can map standardized test entry metrics directly to expected university academic performance thresholds.

This establishes an automated, unbiased screening baseline for incoming applications.

2️⃣ Early Academic Intervention & Counseling
Academic counselors can flag incoming freshmen whose predicted GPAs drop below structural thresholds based on their SAT baseline.

This lets advisors introduce targeted tutoring, mentorship programs, and core skills workshops prior to mid-term drops.

3️⃣ Scholarship & Financial Aid Allocation Optimization
Financial aid committees can simulate combinations of entry qualifications to discover clear performance limits.

This guarantees that merit-based endowment funds are given to candidates with high predicted graduation and retention probabilities.

4️⃣ Institutional Performance & Accreditation Scaling
Predictive models let registrars anticipate department performance shifts and student success rates ahead of time.

This improves university rankings, maintains strict regional accreditation compliance, and optimizes general resource planning across academic years.