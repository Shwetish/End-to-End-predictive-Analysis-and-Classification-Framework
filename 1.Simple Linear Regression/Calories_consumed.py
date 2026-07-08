#=============================================================================
#Business Understanding:-
#=============================================================================
1 Business Problem Statement:
Nutritionists and healthcare professionals want to understand
how daily calorie consumption affects weight gain.
2 Business Objective:
- Identify the strength of relationship between calorie intake
and weight gain
- Build a predictive model for weight gain
- Improve diet planning recommendations
3 Motivation:
- Understanding calorie-weight relationship helps:
- Prevent obesity
- Design controlled diet plans
- Estimate risk of excessive weight gain
4 Constraints:
- Human metabolism varies
- Data size may be small
- Linear assumptions may not always hold
- Outliers may affect regression
5 Success Criteria:
Business Success:
- Accurate prediction of weight gain
- Actionable dietary insights
ML Success:
- Lower RMSE
- Strong correlation
- Stable performance on test data
-----------------------------------------------------------------------
DATA UNDERSTANDING:-
-----------------------------------------------------------------------
'''
Feature Name    Description Type              Business Relevance
wt_gained       Weight gained (grams)         Numeric Target variable
cal_consumed    Calories consumed per day     Numeric Key predictor
'''

# =======================================================================
# CALORIES CONSUMED vs WEIGHT GAINED
# EXPLORATORY DATA ANALYSIS WITH INFERENCE
# =======================================================================
#-------------------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
#-----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------------------
# STEP 2: LOAD THE DATASET
#-----------------------------------------------------------------------
cal = pd.read_csv("C:/21_Linear_Regression_Colory/calories_consumed.csv")
cal.columns = ["wt_gained", "cal_consumed"]

print("First 5 Rows:\n", cal.head())

#-----------------------------------------------------------------------
# STEP 3: BASIC EDA
#-----------------------------------------------------------------------
print("\nData Types:\n", cal.dtypes)
print("\nSummary Statistics:\n", cal.describe())

# -----------------------------------------------------------------------
# BUSINESS MOMENT DECISIONS
# -----------------------------------------------------------------------

# Mean
print("\nMean:\n", cal.mean())
'''
Inference:
• Average calorie intake represents typical daily consumption.
• Average weight gain shows normal gain pattern.
• If both means are high -> population likely in calorie surplus.
• Indicates calories may be contributing to weight gain.
'''
# Variance
print("\nVariance:\n", cal.var())
'''
Inference:
High variance in calories -> different eating habits.
High variance in weight gain -> metabolic differences.
Larger spread means predictions may vary across individuals.
'''

# Standard Deviation
print("\nStandard Deviation:\n", cal.std())
'''
Inference:
Shows average deviation from mean.
Lower value -> stable population behavior.
Higher value -> more fluctuation in diet and weight gain.
'''

# Skewness
print("\nSkewness:\n", cal.skew())
'''
Inference:

Skew ≈ 0 -> Symmetric distribution.
Positive skew -> few individuals consume very high calories.
Positive skew in weight cal consumtion too -> few individuals gain extreme weight.
'''

# Kurtosis
print("\nKurtosis:\n", cal.kurtosis())
'''
Inference:
High kurtosis -> presence of extreme values.
Low kurtosis -> uniform spread.
Extreme calorie or weight values may influence regression.
'''

# Correlation
print("\nCorrelation Matrix:\n",np.corrcoef(cal.cal_consumed, cal.wt_gained))


# -----------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS
# -----------------------------------------------------------------------

# Histogram - Weight Gained
plt.figure(figsize=(6,4))
plt.hist(cal.wt_gained)
plt.title("Weight Gained Distribution")
plt.xlabel("Weight Gained")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
Most individuals gained lower to moderate weight (clustered in the lower 
range)
The distribution appears positively skewed (right-skewed).
A few individuals show very high weight gain, creating a long right tail.
These extreme values may act as outliers and can influence regression results.
'''
# Histogram - Calories Consumed
plt.figure(figsize=(6,4))
plt.hist(cal.cal_consumed)
plt.title("Calories Consumed Distribution")
plt.xlabel("Calories Consumed")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
Calorie intake is spread across a moderate to high range.
Most individuals consume calories within a normal daily intake band.
The distribution shows a slight right skew, indicating a few individuals consume
Presence of high-calorie values suggests potential overconsumption cases.'''

#-----------------------------------------------------------------------
# STEP4:-BOX PLOT (OUTLIER DETECTION)
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=cal[['cal_consumed', 'wt_gained']], orient='h')
plt.title("Boxplot of Numerical Features")

'''
Inference:
Both Calories Consumed and Weight Gained show a reasonable spread.
The data points lie within the whiskers - no extreme outliers detected.
Calories consumed has a wider range compared to weight gained.
Median values appear centrally positioned, indicating stable distributions.
'''

#-----------------------------------------------------------------------
# STEP5:-BIVARIATE ANALYSIS (SCATTER PLOT)
#-----------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='cal_consumed', y='wt_gained', data=cal)
plt.title("calories consumed vs Weight Gained")
plt.xlabel("Calories Consumed")
plt.ylabel("Weight Gained")
plt.show()


'''
Inference:
Clear positive linear relationship observed.
As calories consumed increase, weight gained also increases.
Data points follow an upward trend, indicating a strong correlation.
No major irregular patterns or clustering observed.
'''

#-----------------------------------------------------------------------
# STEP6:-CORRELATION HEATMAP
#--------------------------------------------------------------------------

plt.figure(figsize=(5,4))
sns.heatmap(cal.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

'''
- The correlation between calories consumed and weight gained is 0.95.
- This indicates an exceptionally strong, positive linear relationship.
- In simple terms: as calorie intake increases, weight gain increases significantly.
- Because the value is so close to +1, a Simple Linear Regression model 
  will be highly accurate for predicting calories based on weight gain.
  '''
#-----------------------------------------------------------------------
# PDF & CDF ANALYSIS
#-----------------------------------------------------------------------
for col in ['cal_consumed','wt_gained']:
    plt.figure(figsize=(12,4))
    
    #PDF
    plt.subplot(1,2,1)
    sns.kdeplot(cal[col], fill=True)
    plt.title(f'PDF of {col}')
    
    #CDF
    plt.subplot(1,2,2)
    sorted_vals = np.sort(cal[col])
    y = np.arange(len(sorted_vals))/len(sorted_vals)
    plt.plot(sorted_vals, y)
    plt.title(f'CDF of{col}')
    
    plt.show()
    
'''
1.PDF (Probability Density Function) Analysis:
- The distribution of weight gain is positively skewed (right-skewed).
- Most individuals gained weight in the lower to moderate range 
  (peaking around 200).
- A smaller number of individuals show exceptionally high weight gain,
  forming the right tail.
- This indicates that extreme weight gain cases are limited but present in 
  the data.

2.CDF (Cumulative Distribution Function) Analysis:
- The curve rises sharply early on, showing that a huge chunk of the 
population lies in the lower range.
- Approximately 70% of individuals experienced a weight gain of less than 
600 units.
- Only a tiny percentage of individuals experienced very high weight gain 
(above 800 units), as shown by the flattening curve at the top.
'''
# -----------------------------------------------------------------------
# FINAL EDA SUMMARY
# -----------------------------------------------------------------------
"""
FINAL SUMMARY:-
Strong positive relationship observed between calorie intake 
Distribution mostly stable with few extreme cases.
Correlation confirms calories significantly impact weight.
Linear regression is appropriate for modeling.
Useful for diet planning and obesity prevention strategies.
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
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from feature_engine.outliers import Winsorizer

#-----------------------------------------------------------------------
# STEP 2: LOAD DATASET
#-----------------------------------------------------------------------
cal = pd.read_csv("C:/21_Linear_Regression_Colory/calories_consumed.csv")
cal.columns = ["wt_gained", "cal_consumed"]

print("Initial Shape:", cal.shape)
print(cal.head())

#----------------------------------------------------------------------
#Step 3: Basic Cleaning
#----------------------------------------------------------------------

print("\n Missing Values Before Treatment:\n", cal.isnull().sum())


'''
Inference:
Dataset contains only numerical variables.
No identifier column present.
If missing values exist -> must be handled.
If zero -> dataset is clean.
'''

#-----------------------------------------------------------------------
# STEP 4: MISSING VALUE TREATMENT 
#-----------------------------------------------------------------------
# Using Median Imputation (robust to skewness)
for col in cal.columns:
    cal[col] = cal[col].fillna(cal[col].median())

print("\nMissing Values After Treatment:\n", cal.isnull().sum())

'''
Inference:
-Removes repeated observations.
-Prevents model from learning duplicated patterns.
-Improves generalization capability.
'''    

#-----------------------------------------------------------------------
# STEP 5: DUPLICATE REMOVAL 
#-----------------------------------------------------------------------
cal.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:", cal.shape)

'''
Inference:
Removes repeated observations.
Prevents model from learning duplicated patterns.
Improves generalization capability.
'''

#-----------------------------------------------------------------------
# STEP 6: OUTLIER DETECTION
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=cal, orient='h')
plt.title("Boxplot Before Treatment")
plt.show()
'''
Inference:
- No Outliers Detected: There are no individual data points (dots) plotted 
  outside the "whiskers" on either side of the boxplots.
- Clear Scale Difference: The values for 'cal_consumed' operate on a much 
  higher numeric scale (ranging from ~1,400 to 4,000) compared to 'wt_gained' 
  (ranging from ~0 to 1,100).
- Normal Variation: Both features show clean distributions without extreme 
  anomalies, meaning the dataset is ready for linear regression modeling 
  without requiring outlier treatment.
'''

#-----------------------------------------------------------------------
# STEP 7: OUTLIER TREATMENT (WINSORIZATION)
#-----------------------------------------------------------------------
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['cal_consumed', 'wt_gained']
)

cal[['cal_consumed', 'wt_gained']] = winsor.fit_transform(
    cal[['cal_consumed', 'wt_gained']]
)

plt.figure(figsize=(8,5))
sns.boxplot(data=cal, orient='h')
plt.title("Boxplot After Winsorization")
plt.show()

'''
Inference:
Extreme values capped using IQR method.
'''
#-----------------------------------------------------------------------
# STEP 8: SKEWNESS CHECK
#-----------------------------------------------------------------------
print("\nSkewness:\n", cal.skew())

'''
Inference:
Skewness > 1 -> Strong skew (Log transformation may help).
Mild skew -> transformation optional.
Helps decide model transformation stage.
'''

#-----------------------------------------------------------------------
# STEP9:-FINAL PREPROCESSING SUMMARY
#-----------------------------------------------------------------------
""""
FINAL DATA PREPROCESSING SUMMARY:

• Dataset validated and cleaned.
• Missing values handled using median.
• Duplicates removed.
• Outliers treated using IQR-based Winsorization.
"""
#------------------------------------------------------------------------
# MODEL DEVELOPMENT - SIMPLE LINE LINEAR REGRESSION
# -----------------------------------------------------------------------
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =======================================================================
# 1 SIMPLE LINEAR REGRESSION
# =======================================================================
model1 = smf.ols('wt_gained ~ cal_consumed', data=cal).fit()
pred1 = model1.predict(cal)

rmse1 = np.sqrt(np.mean((cal.wt_gained - pred1)**2))
print("SLR RMSE:", rmse1)
#103.30
model1.summary()
'''

#R-squared = 0.897 > 0.80, Model is very strong
#p = 0 < 0.05 hence acceptable
#beta-0 = -625.75
#beta-1 = 0.4202

Goal of the Model
We are trying to predict wt_gained (Dependent Variable)
using cal_consumed (Independent Variable).
1. Model Fit (Goodness of Fit)
R-squared = 0.897(applicable for MLR)
"Adjusted R² tells us whether adding more variables truly improves the model o
→ About 89.7% of the variation in wt_gained is explained by cal_consumed.
This means the model fits the data very well.
Adjusted R-squared = 0.888
→ Adjusted for number of predictors; still very high.
Since there is only one predictor, overfitting is not a concern.

purpose of F-statistic
Does the independent variable(s) collectively explain the dependent variable?
Or is the relationship happening just by chance?
F-statistic = 104.3
An F-value of 104.3 is very large.
This means the model explains much more variation in weight gain than random c
It indicates that calories consumed significantly improves prediction of weight

Model is statistically significant.

3. Residual Analysis

The Durbin-Watson statistic checks whether residuals are independent.

DW Value Meaning
≈ 2   No autocorrelation (Ideal)
< 1.5      Positive autocorrelation
> 2.5      Negative autocorrelation
1.5-2.5    Acceptable

Durbin-Watson = 2.537

Slightly above 2 but still acceptable.
No serious autocorrelation problem.

Normality Check

Jarque-Bera p-value = 0.541 (> 0.05)
Residuals are approximately normally distributed.
Regression assumption satisfied.
'''
# =======================================================================
# 2 LOG MODEL (log X)
# =======================================================================
model2 = smf.ols('wt_gained ~ np.log(cal_consumed)', data=cal).fit()
pred2 = model2.predict(cal)

rmse2 = np.sqrt(np.mean((cal.wt_gained - pred2)**2))
print("Log-X Model RMSE:", rmse2)
#141.005
model2.summary()

'''

#R-squared = 0.808 = 0.8, there is scope for improvement
#p = 0.000 < 0.05 hence acceptable
#beta-0 = -6955.65
#beta-1 = 948.37

Goal of the Model
We are trying to predict wt_gained (Dependent Variable) using log(cal_consume

Model Fit (Goodness of Fit)

F-statistic = 50.40
Prob(F) = 1.25e-05 (< 0.05)

Model is statistically significant.
  The log transformation of calories significantly explains weight gain.
  The model performs much better than a model with no predictor.

Coefficient Interpretation

Intercept = -6955.65
When log(calories) = 0, predicted weight gain is -6955.65 (not practically mea

log(cal_consumed) = +948.37
For every 1 unit increase in log(calories), weight increases by 948.37 units.
  Strong positive relationship.

P-value for log(cal_consumed) = 0.000
  Highly statistically significant.

This confirms:
Higher calorie intake (even after log transformation) leads to higher weight g

Residual Analysis

Durbin-Watson = 2.438
→ Lies within acceptable range (1.5-2.5).
→ No serious autocorrelation problem.

Normality Check

Jarque-Bera p-value = 0.566 (> 0.05)
  Residuals are approximately normally distributed.
  Regression assumptions are satisfied.

Final Comparison Insight

Although the model is statistically significant and explains about 81% of the 

In One Sentence :

"Log(calories) has a statistically significant positive impact on weight gaine
'''

# =======================================================================
# 3 EXPONENTIAL MODEL (log Y)
# =======================================================================
model3 = smf.ols('np.log(wt_gained) ~ cal_consumed', data=cal).fit()
pred3 = model3.predict(cal)

# since model is in log(Y), we need to take exponential to convert back to 
pred3_back = np.exp(pred3)

rmse3 = np.sqrt(np.mean((cal.wt_gained - pred3_back)**2))
print("Exponential Model RMSE:", rmse3)

'''

#R-squared = 0.878 > 0.80, Model is strong
#p = 0.00 < 0.05 hence acceptable
#beta-0 = 2.8387
#beta-1 = 0.0011

Goal of the Model
We are trying to predict log(wt_gained) (Dependent Variable) using cal_consume

Model Fit (Goodness of Fit)

R-squared = 0.878
→ About 87.8% of the variation in log(wt_gained) is explained by cal consumed.
'''
# =======================================================================
# 4 POLYNOMIAL MODEL
# =======================================================================
model4 = smf.ols(
    'np.log(wt_gained) ~ cal_consumed + I(cal_consumed**2)',
    data=cal
).fit()

pred4 = np.exp(model4.predict(cal))
rmse4 = np.sqrt(np.mean((cal.wt_gained - pred4)**2))
print("Polynomial Model RMSE:", rmse4)
#117.41
model4.summary()

'''
#R-squared = 0.878 > 0.85, Model is strong
#Adjusted R-squared = 0.855 (slightly reduced after adding extra term)
#p (F-statistic) = 9.61e-06 < 0.05 hence overall model acceptable
#beta-0 = 2.8287
#beta-1 = 0.0011
#beta-2 = -1.675e-09

Goal of the Model

The polynomial term does NOT improve the model.
The relationship remains primarily linear.

Residual Analysis

Durbin-Watson = 3.131
  Greater than 2.5
  Possible negative autocorrelation present.

Normality Check

Jarque-Bera p-value = 0.0466 (< 0.05)
  Residuals are not perfectly normally distributed.

Final Interpretation

Although the polynomial model has high R² (0.878), the squared term is complet

In One Sentence :

"The polynomial model explains about 88% of the variation in log(weight gainea
'''
# =======================================================================
# 5.MODEL COMPARISON
# =======================================================================
results = pd.DataFrame({
    "Model": ["SLR", "Log-X", "Exponential", "Polynomial"],
    "RMSE": [rmse1, rmse2, rmse3, rmse4],
    "R_squared": [
        model1.rsquared,
        model2.rsquared,
        model3.rsquared,
        model4.rsquared
    ]
})

print("\nModel Comparison:\n", results)

# =======================================================================
#6. SELECT BEST MODEL
# =======================================================================
# Create the results summary table (add your models here as you complete them)
results_data = {
    "Model": ["Simple Linear Regression (Base)"],
    "RMSE": [rmse1]  # This is the rmse1 variable you calculated in Model 1 (103.30)
}

# Define the 'results' DataFrame that was missing
results = pd.DataFrame(results_data)

# Now your sorting code will run perfectly without errors!
best_model_name = results.sort_values("RMSE").iloc[0]["Model"]

# Print the best model
print(f"\nBest Model: {best_model_name}")

# =======================================================================
# 7.VISUALIZE FINAL MODEL FITS
# =======================================================================
plt.figure(figsize=(12, 10))

# Scatter plot of original data
plt.scatter(
    cal.cal_consumed,
    cal.wt_gained,
    color='blue',
    label='Actual Data',
    alpha=0.7
)

# Plot fitted lines
plt.plot(cal.cal_consumed, pred1, color='red', label=f'SLR (R²={model1.rsquared:.3f})')
#==========================================================================
# 8.SELECT BEST MODEL
# =======================================================================
best_model_name = results.sort_values("RMSE").iloc[0]["Model"]
print("\nBest Model Based on RMSE:", best_model_name)

# -----------------------------------------------------------------------
# 9.TRAIN-TEST VALIDATION USING BEST MODEL (SLR IS BEST HERE)
# -----------------------------------------------------------------------

train, test = train_test_split(cal, test_size=0.3, random_state=42)

# Since SLR has highest R² (0.897) and lowest RMSE, choose model1
final_model = smf.ols('wt_gained ~ cal_consumed', data=train).fit()

train_pred = final_model.predict(train)
test_pred = final_model.predict(test)

train_rmse = np.sqrt(np.mean((train.wt_gained - train_pred)**2))
test_rmse = np.sqrt(np.mean((test.wt_gained - test_pred)**2))

print("\nTrain RMSE:", train_rmse)
print("Test RMSE :", test_rmse)

#===================================================================
#10.Business Impact of the Project
#===================================================================
Calories Consumed vs Weight Gained Analysis
1️ Data-Driven Diet Planning
The model shows a strong positive relationship between calorie intake and weight gain.
About 80–90% of weight variation is explained by calories.
Enables nutritionists to predict expected weight gain for a given calorie intake.
Helps design personalized calorie-controlled diet plans.

2️ Obesity Risk Identification
Individuals consuming higher calories are statistically more likely to gain more weight.
High-risk calorie thresholds can be identified using CDF analysis.
Supports early intervention programs for obesity prevention.

3️ Preventive Healthcare Strategy
Healthcare providers can estimate:
How much weight gain may occur
What calorie limit keeps weight stable
Reduces risk of:
Diabetes
Hypertension
Cardiovascular diseases

4️ Fitness & Wellness Industry Application
Gyms and wellness apps can:
Predict weight gain/loss trends
Provide calorie recommendations
Personalize fitness targets
