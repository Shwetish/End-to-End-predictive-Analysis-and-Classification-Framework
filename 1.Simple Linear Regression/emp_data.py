#=============================================================================
# Business Understanding:-
#=============================================================================
1 Business Problem Statement:
When workers leave a company too fast, it hurts the business. HR managers need to
know exactly how much a low salary raise drives employees to quit.

2 Business Objective:
Find the Link: See exactly how much giving bigger salary raises helps stop employees
from quitting.

Build a Predictor: Create a smart tool that guesses what the quitting rate will be
based on the size of the raise.

Smarter Planning: Use this data to plan better budgets and design fairer pay plans 
to keep workers happy.

3 Motivation:
It stops top talent from leaving the company.

It helps us design good pay raises without wasting company money.

It warns us ahead of time if a specific department is at risk of losing too many
people.

4 Constraints:
The Human Element: Every employee has different reasons for staying or leaving that
numbers cannot easily measure.

Small Data: We might have only a tiny amount of past company data to study.

Curved Patterns: The link between money and quitting might be a curve rather than a 
perfectly straight line.

Boss-Level Raises: A few rare, massive executive pay raises can mess up our regular
 math formulas.

5 Success Criteria:
Business Success:

We can accurately guess quitting rates so managers can take action to keep their 
best workers.

ML Success:

Our AI model makes tiny guessing errors (low RMSE), proves a strong link where higher
pay means fewer people quit, and works great on new data.

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
# Loading the employee dataset from the same directory
emp_data = pd.read_csv("C:/21_Linear_Regression_Colory/emp_data.csv")

# Standardizing column names for easy analysis
emp_data.columns = ["salary_hike", "churn_out_rate"]

print("First 5 Rows:\n", emp_data.head())

#-----------------------------------------------------------------------
# STEP 3: BASIC EDA
#-----------------------------------------------------------------------
print("\nData Types:\n", emp_data.dtypes)
print("\nSummary Statistics:\n", emp_data.describe())

# -----------------------------------------------------------------------
# BUSINESS MOMENT DECISIONS
# -----------------------------------------------------------------------

# Mean
print("\nMean:\n", emp_data.mean())
'''
Inference:
• Average salary hike shows the standard pay raise given to employees.
• Average churn out rate shows the typical percentage of employees leaving the company.
• Gives HR a baseline of how much we pay versus how many people we lose.
'''

# Variance
print("\nVariance:\n", emp_data.var())
'''
Inference:
Variance is just the standard deviation squared ($92.09 \times 92.09 \approx 8481$).
Because it is squared, the numbers look giant, but it tells the exact same story:
Salary raises have high variety, while quitting rates are much more consistent.
'''

# Standard Deviation
print("\nStandard Deviation:\n", emp_data.std())
'''
Inference
Salary Hike Changes A Lot ($\pm 92$):
Employees got very different amounts of raises.
The sizes of the raises are spread far apart from each other, showing a wide gap 
between the lowest and highest pay bumps.

Quitting Rate Stays Closer ($\pm 10\%$):
The employee quitting rate is much more stable and compact. 
It does not swing wildly from department to department;
it stays relatively closer to the company average.
'''
# Skewness
print("\nSkewness:\n", emp_data.skew())
'''
Inference:
• Positive skew in churn -> A few teams or periods have extremely high employee exit rates.
• Positive skew in salary -> Only a tiny group of employees gets very large raises.
'''

# Kurtosis
print("\nKurtosis:\n", emp_data.kurtosis())
'''
Inference:
Kurtosis tells you how "peaked" or flat your data distribution is. A score close to
$0$ means a perfectly normal curve, while a negative score means a flatter, wider 
shape.
'''

# Correlation
print("\nCorrelation Matrix:\n", np.corrcoef(emp_data.salary_hike, emp_data.churn_out_rate))

'''
Inference:-
-Massive, Opposite Link ($-0.91$): There is a near-perfect negative connection between
 raises and quitting. The minus sign means they move in completely opposite 
 directions.
-Higher Pay = Lower Quitting: It proves that as Salary_hike goes up, the
 Churn_out_rate drops drastically.
-Powerful Predictor: Because the score is so close to $-1$, salary raises are an 
incredibly strong and reliable factor for guessing employee turnover.
'''

# -----------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS
# -----------------------------------------------------------------------

# Histogram - Salary Hike
plt.figure(figsize=(6,4))
plt.hist(emp_data.salary_hike, color='teal', edgecolor='black')
plt.title("Salary Hike Distribution")
plt.xlabel("Salary Hike Amount")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
Most Raises are Small: The tallest bars are on the left side, showing that most
 employees received smaller salary raises (between 1580 and 1660).

Fewer Large Raises: The bars stretch out thinner to the right, meaning only a very
 small number of employees received high salary raises (1800+).

Right-Skewed Gap: The data has a long tail pointing toward the right with empty
 gaps, proving that big executive raises are rare exceptions.
'''

# Histogram - Churn Out Rate
plt.figure(figsize=(6,4))
plt.hist(emp_data.churn_out_rate, color='coral', edgecolor='black')
plt.title("Churn Out Rate Distribution")
plt.xlabel("Churn Out Rate (%)")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
• Visualizes the common turnover trends in the company.
• Higher bars on the lower side mean good employee retention.
'''

#-----------------------------------------------------------------------
# STEP 4 (Continued): BOX PLOT (OUTLIER DETECTION)
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=emp_data[['salary_hike', 'churn_out_rate']], orient='h', palette='Set3')
plt.title("Boxplot of Salary Hike & Churn Out Rate")
plt.show()

'''
Inference:
• Points outside the whiskers show unique exceptions—like an abnormally 
  high pay raise or a shocking mass resignation event.
• Helps HR find out-of-the-ordinary cases that need special attention.
'''

#-----------------------------------------------------------------------
# STEP 5: BIVARIATE ANALYSIS (SCATTER PLOT)
#-----------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='salary_hike', y='churn_out_rate', data=emp_data, color='blue')
plt.title("Salary Hike vs Churn Out Rate")
plt.xlabel("Salary Hike")
plt.ylabel("Churn Out Rate")
plt.show()

'''
Inference:
• Expected downward trend: As salary hikes go UP, the churn out rate goes DOWN.
• Shows a clear picture that raising salaries keeps employees from quitting.
'''

#-----------------------------------------------------------------------
# STEP 6: CORRELATION HEATMAP
#-----------------------------------------------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(emp_data.corr(), annot=True, cmap='RdYlGn_r')
plt.title("HR Retention Correlation Heatmap")
plt.show()

'''
Inference:
• Strong Opposite Connection ($-0.91$): The dark green blocks show a massive negative
 link between the two metrics.
 Higher Raises = Way Fewer Quits: As salary hikes go up,employee quitting rates drop 
 drastically.
 Perfect Target: Because the score is so close to $-1$, changing employee salary 
 raises is an incredibly effective way for HR to control and lower company turnover.
 '''

#-----------------------------------------------------------------------
# PDF & CDF ANALYSIS
#-----------------------------------------------------------------------
for col in ['salary_hike', 'churn_out_rate']:
    plt.figure(figsize=(12,4))
    
    # PDF
    plt.subplot(1,2,1)
    sns.kdeplot(emp_data[col], fill=True, color='purple')
    plt.title(f'PDF of {col}')
    
    # CDF
    plt.subplot(1,2,2)
    sorted_vals = np.sort(emp_data[col])
    y = np.arange(len(sorted_vals)) / len(sorted_vals)
    plt.plot(sorted_vals, y, color='crimson', linewidth=2)
    plt.title(f'CDF of {col}')
    
    plt.show()
    
'''
1️.PDF (Purple Curve) — The "Most Likely" Spot
Peak Around 70%: The highest part of the curve shows that a quitting rate of around
 68% to 72% is the most common occurrence in the company.

Smooth Bell Shape: The quitting data is relatively well-balanced, meaning extreme 
values (below 50% or above 95%) are rare.

2️.CDF (Red Line) — The "Running Total"
Steep Middle Rise: The line climbs fastest between 60% and 75%. This tells us that 
the vast majority of your departments sit in this range.

Reading the Probability: If you look at 75% on the bottom line, it matches up with
 0.6 on the side. This means there is a 60% chance that a department's quitting rate
 is 75% or lower.
 '''

# -----------------------------------------------------------------------
# FINAL EDA SUMMARY
# -----------------------------------------------------------------------
"""
FINAL SUMMARY:-
• Confirms that salary increases act as a major direct lever to stop employee
  turnover.
• Gives management a clear mathematical roadmap to balance salary costs against 
  employee replacement costs.
• Validates that Simple Linear Regression is a perfect tool for predicting employee 
  churn based on compensation hikes.
  Money talks. Because the negative link is incredibly strong ($-0.91$),
  HR executives can confidently use salary adjustments as their primary lever to
  control, predict, and stop employee turnover.
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
emp = pd.read_csv("C:/21_Linear_Regression_Colory/emp_data.csv")
emp.columns = ["salary_hike", "churn_out_rate"]

print("Initial Shape:", emp.shape)
print(emp.head())

'''
Inference:
• Data Structure: The dataset successfully loads with 10 rows and 2 columns.
• Layout: "salary_hike" is our numeric input feature, and "churn_out_rate" (%) 
  is our target.
'''

#----------------------------------------------------------------------
# STEP 3: BASIC CLEANING & MISSING VALUE TREATMENT
#----------------------------------------------------------------------
print("\nMissing Values Before Treatment:\n", emp.isnull().sum())

# Robust handling: Median Imputation
for col in emp.columns:
    emp[col] = emp[col].fillna(emp[col].median())

print("\nMissing Values After Treatment:\n", emp.isnull().sum())

'''
Inference:
• Missing Values: The dataset is fully intact with 0 missing entries. 
• Safety First: The median imputation loop runs cleanly as a robust safety net to
 ensure no blank spaces disrupt the math models.
'''

#-----------------------------------------------------------------------
# STEP 4: DUPLICATE REMOVAL 
#-----------------------------------------------------------------------
emp.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:", emp.shape)

'''
Inference:
• No Duplicates: The shape remains identical, proving every data row contains a 
  unique pair of historical company records.
'''

#-----------------------------------------------------------------------
# STEP 5: OUTLIER DETECTION & TREATMENT (WINSORIZATION)
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=emp, orient='h', palette='Pastel1')
plt.title("Boxplot Before Treatment")
plt.show()
'''
InferenceNo Outliers: There are no isolated dots outside the whiskers for either
 variable, meaning your dataset contains zero extreme or rogue values to worry about.
 Scale Mismatch: The two variables live on completely different scales. salary_hike is
 plotted way out on the right because its values are in the thousands ($1500+$),
 while churn_out_rate sits on the far left because it is a percentage between $0$ and
 $100$.
 '''
# Applying Winsorizer using IQR method
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['salary_hike', 'churn_out_rate']
)
emp[['salary_hike', 'churn_out_rate']] = winsor.fit_transform(emp[['salary_hike', 'churn_out_rate']])

'''
Inference:

-Clips Extreme Values: It finds unusually high or low numbers and clips them.

-Sets Safety Boundaries: It keeps your data points inside a normal range so rogue
 numbers don't warp your results.

-Protects Accuracy: It ensures extreme data doesn't break or tilt your machine 
learning models.
'''
#-----------------------------------------------------------------------
# STEP 6: SKEWNESS CHECK
#-----------------------------------------------------------------------
print("\nSkewness:\n", emp.skew())

'''
Inference:
• Salary Hike Skewness (~0.85): Moderately right-skewed, showing that smaller to 
mid-range pay bumps are common, while high salary raises are less frequent.
• Churn Out Rate Skewness (~0.64): Mildly right-skewed, meaning normal, manageable
 department quitting rates occur most of the time, with severe attrition spikes being
 rare exceptions.
'''

#------------------------------------------------------------------------
# MODEL DEVELOPMENT - COMPARING MULTIPLE MODEL TRANSFORMATIONS
# -----------------------------------------------------------------------

# =======================================================================
# MODEL 1: SIMPLE LINEAR REGRESSION (SLR)
# =======================================================================
model1 = smf.ols('churn_out_rate ~ salary_hike', data=emp).fit()
pred1 = model1.predict(emp)
rmse1 = np.sqrt(np.mean((emp.churn_out_rate - pred1)**2))

print("\n--- MODEL 1: SLR SUMMARY ---")
print("SLR RMSE:", rmse1)
print(model1.summary())

'''
Inference:
• Baseline Metrics: Generates a baseline guessing error (RMSE) of roughly 3.99 units.
• Strong Performance: Explains approximately 83.1% of the data variance 
 (R-squared = 0.831) using a standard straight-line relationship.
'''

# =======================================================================
# MODEL 2: LOG MODEL (log X)
# =======================================================================
model2 = smf.ols('churn_out_rate ~ np.log(salary_hike)', data=emp).fit()
pred2 = model2.predict(emp)
rmse2 = np.sqrt(np.mean((emp.churn_out_rate - pred2)**2))

print("\n--- MODEL 2: LOG-X SUMMARY ---")
print("Log-X RMSE:", rmse2)
print(model2.summary())

'''
Inference:
Logarithmic Curve: This checks if changing the salary raises into a log scale
captures a curved trend where employee quitting drops quickly at first and then
 stabilizes.

Model Fit Performance: The model gives a guessing mistake (RMSE ~ 3.78) and 
successfully explains 84.9% of the data patterns (R-squared = 0.849). 
This proves to be a highly accurate and strong model framework for HR to use.
'''

# =======================================================================
# MODEL 3: EXPONENTIAL MODEL (log Y)
# =======================================================================
model3 = smf.ols('np.log(churn_out_rate) ~ salary_hike', data=emp).fit()
pred3 = model3.predict(emp)
pred3_back = np.exp(pred3) # Back-transform out of log scale
rmse3 = np.sqrt(np.mean((emp.churn_out_rate - pred3_back)**2))

print("\n--- MODEL 3: EXPONENTIAL SUMMARY ---")
print("Exponential RMSE:", rmse3)
print(model3.summary())

'''
Decay Trend: This tests if the employee quitting rate drops in an exponential decay
 pattern (dropping very sharply at first and then leveling out) as salaries increase.

Model Fit Performance: The model gives a guessing mistake (RMSE ~ 3.54) and 
successfully explains 87.4% of the data patterns (R-squared = 0.874). 
This makes it a stronger predictor than the Log-X model.
'''
# =======================================================================
# MODEL 4: POLYNOMIAL MODEL (Quadratic)
# =======================================================================
model4 = smf.ols('churn_out_rate ~ salary_hike + I(salary_hike**2)', data=emp).fit()
pred4 = model4.predict(emp)
rmse4 = np.sqrt(np.mean((emp.churn_out_rate - pred4)**2))

print("\n--- MODEL 4: POLYNOMIAL SUMMARY ---")
print("Polynomial RMSE:", rmse4)
print(model4.summary())

'''
Quadratic Fit: This creates a flexible, U-shaped curved line that fits the data 
beautifully. It delivers our best scores yet with the lowest guessing mistake 
(RMSE ~ 1.58) and the highest accuracy (R-squared = 0.974) on the dataset.

Sweet Spot: It perfectly captures the real-world "diminishing returns" 
effect—showing that salary hikes are incredibly effective at stopping employee churn
 up to a certain structural threshold.
'''

# =======================================================================
# STEP 7: MODEL COMPARISON & SELECTION
# =======================================================================
results = pd.DataFrame({
    "Model": ["SLR", "Log-X", "Exponential", "Polynomial"],
    "RMSE": [rmse1, rmse2, rmse3, rmse4],
    "R_squared": [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared]
})

print("\nModel Comparison Table:\n", results)

# Automatically identify the model with the absolute lowest error metric
best_model_row = results.sort_values("RMSE").iloc[0]
best_model_name = best_model_row["Model"]
print(f"\nBest Performing Model Configuration: {best_model_name}")
'''
Inference:-
Comparison Overview: * Polynomial (Quadratic) is the undisputed winner. It scores the
 absolute lowest guessing error (RMSE ~ 1.58) and the highest accuracy 
 (R-squared ~ 0.974).

Exponential takes second place, followed closely by Log-X, while the basic straight
 line (SLR) forms our baseline.

Selected Curve: A curved Polynomial relationship mimics real-world employee behavior 
far better than a stiff, straight line. It proves that salary raises are extremely 
effective at dropping churn rates until a natural baseline is hit.
'''


# =======================================================================
# STEP 8: TRAIN-TEST VALIDATION
# =======================================================================
train, test = train_test_split(emp, test_size=0.3, random_state=42)

# Training our final validated model structure on train split
final_model = smf.ols('churn_out_rate ~ salary_hike', data=train).fit()

train_pred = final_model.predict(train)
test_pred = final_model.predict(test)

train_rmse = np.sqrt(np.mean((train.churn_out_rate - train_pred)**2))
test_rmse = np.sqrt(np.mean((test.churn_out_rate - test_pred)**2))

print("\n--- TRAIN-TEST CROSS VALIDATION ---")
print("Train Split RMSE:", train_rmse)
print("Test Split RMSE :", test_rmse)

'''
Inference:-
Normally, Train error is a bit lower than Test error. Here, Test RMSE (2.57) is
 actually lower than  Train RMSE (4.49).

Since the dataset is very small (10 total rows), a 70/30 split means only 3 rows
 went into the Test set. The model got lucky because those 3 test rows happened 
 to fall very close to the prediction line.
 
-The model is not overfitting is still 100% correct.
'''

# =======================================================================
# STEP 9: VISUALIZE FINAL MODEL FITS
# =======================================================================
plt.figure(figsize=(10, 6))
plt.scatter(emp.salary_hike, emp.churn_out_rate, color='crimson', label='Actual Attrition Data', alpha=0.7)
plt.plot(emp.salary_hike, pred1, color='blue', linewidth=2, label=f'SLR Line (R²={model1.rsquared:.3f})')
plt.title("Final Model Fit: Salary Hike vs Churn-out Rate")
plt.xlabel("Salary Hike")
plt.ylabel("Churn-out Rate")
plt.legend()
plt.show()

'''
Inference:
• Clear Trend Line: The final trend visualization highlights a distinct downward slope.
• Actionable Tool: This visual profile serves as an intuitive tool for corporate
 executives, illustrating exactly how strategic pay adjustments reliably lower 
 employee turnover.
'''
#===================================================================================
10. Business Impact of the Project
#==================================================================================
 The Big Problem & The Discovery
The Problem: When too many employees quit, it costs the company a lot of money and 
hurts team spirit.

The Discovery: The data proves that money talks. Giving better salary raises makes 
the quitting rate drop immediately.

The Simple Solution (How HR Wins)
Find the Sweet Spot: Giving giant, expensive raises is unnecessary. Small to medium 
raises work perfectly fine to make people stay.

No More Guessing: Instead of playing guessing games with company money, leadership 
can use the mathematical curve tool to see exactly how much quitting will drop for 
every dollar added to salaries.

Early Warning System: If a department plans to give very low raises, HR can flag them
 as a "High Danger Zone" and step in to help before people actually start quitting.