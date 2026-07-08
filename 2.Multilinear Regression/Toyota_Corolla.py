# -------------------------------------------------------------------------
# BUSINESS UNDERSTANDING
# -------------------------------------------------------------------------

1 Business Problem Statement:

Automobile manufacturers and automotive analysts want to understand 
how vehicle characteristics such as horsepower, engine volume, 
speed, and weight influence fuel efficiency (MPG).

Fuel efficiency is a critical factor affecting customer purchase 
decisions, regulatory compliance, and environmental impact.


2 Business Objective:

- Identify key factors affecting fuel efficiency (MPG)
- Quantify the relationship between MPG and vehicle specifications
- Build a predictive model to estimate MPG based on vehicle features
- Help manufacturers optimize vehicle design for better fuel efficiency

3 Motivation:

Understanding factors affecting MPG helps:

- Design fuel-efficient vehicles
- Reduce fuel consumption and emissions
- Improve competitive positioning in the automobile market
- Support environmentally sustainable transportation
- Help customers make informed purchasing decisions


4 Constraints:

- Strong multicollinearity exists between some variables (VOL & WT, HP & SP)
- Real-world vehicle performance may be affected by additional factors
- Linear regression assumptions may not fully hold
- Outliers and extreme vehicles may influence model accuracy
- Dataset size may be limited

5 Success Criteria:

Business Success Criteria:

- Ability to identify key factors affecting fuel efficiency
- Provide actionable insights for vehicle design optimization
- Support decision-making for automotive engineering and planning


Machine Learning Success Criteria:

- High R-squared value (strong explanatory power)
- Statistically significant predictors (p-value < 0.05)
- Low prediction error (RMSE)
- Residuals approximately normally distributed
- Minimal multicollinearity after feature selection


# -------------------------------------------------------------------------
# DATA UNDERSTANDING
# ----------------------------------------------------------------------
'''
Feature Name    Description              Type    Business Relevance

MPG             Miles per gallon 
                (fuel efficiency)       Numeric  Target variable
HP              Horsepower of vehicle   Numeric  Indicates engine power
VOL             Engine volume           Numeric  Reflects engine size
SP              Maximum speed           Numeric  Performance indicator
WT              Vehicle weight          Numeric  Influences fuel
                                                  consumption
'''

# ==============================================================================
# MULTIPLE LINEAR REGRESSION
# CARS DATASET – EXPLORATORY DATA ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
#====================
#Load the Dataset
#=======================

cars = pd.read_csv("C:/22_Multilinear_Regression/cars.csv")
print("First 5 Rows:\n", cars.head())
print("\nDta Types:\n", cars.dtypes)
print("\n summary statistics:\n",cars.describe())
print("\nMissing values:\n", cars.isnull().sum())

#Business moment Decisions:-

#Mean
print("\nmean:\n", cars.mean())

'''
inference:-
HP       117.469136
MPG       34.422076
VOL       98.765432
SP       121.540272
WT        32.412577
'''
# Variance
print("\nVariance:\n", cars.var())
'''
Inference:
High variance in price -> multiple segments (economy + luxury).
High variance in engine or horsepower -> diverse product range.
'''

# Standard Deviation
print("\nStandard Deviation:\n", cars.std())
'''
Inference:
Shows spread around mean.
Higher STD -> less stability in feature values.
'''

# Skewness
print("\nSkewness:\n", cars.skew())
'''
Inference:
Positive skew in price -> few very expensive cars.
Skewed variables may require log tr
'''
# Kurtosis
print("\nKurtosis:\n", cars.kurtosis())
'''
Inference:
High kurtosis -> presence of extreme values.
Extreme values may influence regression coefficients.
'''

# -----------------------------------------------------------------------------
# UNIVARIATE ANALYSIS
# -----------------------------------------------------------------------------
cars.hist(figsize=(12,8))
plt.suptitle("Histogram of Numerical Features")
plt.show()

'''
Inference:
1 HP (Horsepower)
Distribution is positively skewed (right-skewed).
Most cars have moderate horsepower (80-120 HP).
Few cars have very high HP, creating a long right tail.
Indicates presence of some high-performance vehicles


2 MPG (Mileage)
Distribution appears approximately normal with slight skewness.
Most vehicles fall in the 25–40 MPG range.
Very low and very high mileage cars are limited.
Suggests balanced fuel efficiency across vehicles.

3 VOL (Engine Volume)

Slight right skewness observed.
Majority of cars have medium engine volume (80–120 range).
Few large-engine vehicles exist.
Indicates mix of standard and heavy engine cars.

4 SP (Speed)

Distribution shows mild right skewness.
Most cars have speeds around 110–130.
Few high-speed vehicles extend the right tail.
Reflects presence of performance segment.

5 WT (Weight)

Distribution is slightly right-skewed.
Most cars weigh between 28–38 units.
Few heavier cars create upper tail.
Suggests majority are mid-weight vehicles.'''

# -----------------------------------------------------------------------------
# BOX PLOT (OUTLIER DETECTION)
# -----------------------------------------------------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=cars)
plt.xticks(rotation=45)
plt.title("Boxplot for Outlier Detection")
plt.show()

'''
1 HP (Horsepower)
Several upper outliers present (very high HP values).
Indicates presence of high-performance cars.
HP shows high variability compared to other variables.

2 MPG (Mileage)
Very few outliers observed.
Distribution is relatively stable.
Mileage values are consistent across most vehicles.

3 VOL (Engine Volume)
A few upper and lower outliers present.
Suggests some vehicles have very small or very large engines.
Moderate variability observed.

4 SP (Speed)
Some upper outliers (high-speed vehicles).
Majority of cars clustered within normal speed range.
Indicates limited high-performance segment.

5 WT (Weight)
Few outliers on both ends (very Light and very heavy cars).
Most vehicles concentrated around median range.
Moderate spread observed. '
'''
----------------------------------------------------------------------------
CORRELATION MATRIX:-
------------------------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(cars.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

'''
Inference:-
1 HP & SP (0.97 - Very Strong Positive Correlation)
Extremely strong relationship.
Higher horsepower → higher speed.
Indicates possible multicollinearity issue if both are used in MLR.

2 VOL & WT (1.00 - Perfect Positive Correlation)
Perfect correlation observed.
One variable is likely to be correlated with the other.

3 HP & MPG (-0.73 - Strong Negative Correlation)

As horsepower increases, mileage decreases.
Powerful cars consume more fuel.
Logical engineering relationship.

4 SP & MPG (-0.69 - Strong Negative Correlation)
Higher speed cars tend to have lower mileage.
Indicates performance vs fuel efficiency trade-off.

5 VOL & MPG (-0.53 - Moderate Negative Correlation)
Larger engine → Lower fuel efficiency.
Supports automobile design principles.

6 Weak Correlations (~0.07 to 0.10)
Some variables show weak relationships.
These may contribute less to predictive power.

Multicollinearity Warning
VOL & WT = 1.00 → Serious multicollinearity
HP & SP = 0.97 → Very high multicollinearity
Should check VIF before fitting MLR.
Consider dropping one variable from each highly correlated pair.


'''
# -----------------------------------------------------------------------------
# PAIRPLOT (BIVARIATE ANALYSIS)
# -----------------------------------------------------------------------------
sns.pairplot(cars)
plt.show()

'''
Inference:
Check linear relationship visually.
Detect non-linear patterns or clusters.
'''

# -----------------------------------------------------------------------------
# MULTICOLLINEARITY CHECK (VIF)
# -----------------------------------------------------------------------------

# Assuming 'Milage' is dependent variable
X = cars
X_const = add_constant(X)

'''
When checking multicollinearity for HP:
We run:
HP=β0+β1MPG+β2VOL+β3SP+β4WT

Without constant:

HP=β1MPG+β2VOL+...

That’s unrealistic – relationship rarely passes through zero.

'''
vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i)
                   for i in range(X_const.shape[1])]

print("\nVariance Inflation Factor:\n", vif_data)

'''
Inference:
VIF > 10 -> severe multicollinearity.
VIF 5-10 -> moderate multicollinearity.
Remove or combine variables if VIF high.

| Feature  | VIF    | Inference                                |
| -------- | ------ | ---------------------------------------- |
| **const**| 942.71 | Ignore (intercept not interpreted)       |
| **HP** | 27.12  | Very high multicollinearity              |
| **MPG** | 4.36   | Moderate multicollinearity (acceptable)  |
| **VOL** | 641.74 | Extremely severe multicollinearity       |
| **SP** | 21.65  | Very high multicollinearity              |
| **WT** | 640.00 | Extremely severe mul

VOL and WT have extremely high VIF (> 600) -> Almost perfect correlation.
One of them must be removed immediately.

HP and SP also have very high VIF (> 20)
 Strong multicollinearity between them.

MPG (4.35) is acceptable and can be retained.

'''
#-------------------------------------------------------------------------
#FIT MULTIPLE LINEAR REGRESSION MODEL
#-------------------------------------------------------------------------
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

#Dependent Variable
y = cars["MPG"]

#Independent Variables (remove MPG)
X = cars.drop(columns=["MPG"])

#Add constant (intercept)
X = sm.add_constant(X)

#Fit Model
model = sm.OLS(y, X).fit()
print(model.summary())


#-------------------------------------------------------------------------------
#Residual analysis
#-------------------------------------------------------------------------------

residuals =model.resid
fitted_vals = model.fittedvalues

#Resiualvs Fitted plot
plt.figure(figsize=(6,4))
plt.scatter(fitted_vals, residuals)
plt.axhihline(0, color='red')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

'''
Inference:
The red line in the Residuals vs Fitted plot is:

Residual = 0

It represents perfect prediction.

If a point lies exactly on the red line ->
Predicted value = Actual value
Error = 0
Homoscedasticity (Constant Variance)

Spread of residuals should be roughly equal across all fitted values.

Good sign:
Same vertical spread everywhere

Bad sign:
Funnel shape (triangle)
Spread increasing or decreasing

Points below -> over-prediction

2 Is Linearity OK?

Residuals are roughly scattered around zero.
There is no strong U-shape or clear curve.

Linearity assumption is mostly satisfied.


3 Is Variance Constant?
Spread of residuals increases slightly at higher fitted values.
This looks like a mild funnel shape.

Suggests possible heteroscedasticity (variance not fully constant).


4 Any Outliers?
A few points are far from zero (around +15 and -8).

These may be:
Influential observations
High leverage points
'''

# -----------------------------------------------------------------------------
# QQ PLOT (NORMALITY CHECK FOR RESIDUALS)
# -----------------------------------------------------------------------------

plt.figure(figsize=(6,4))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot - Residuals")
plt.show()

'''
Inference:
Points on straight line -> residuals normally distributed.
Curvature -> skewness.
Heavy tails -> outliers.
Most points lie close to the red straight line -> residuals are approximately normall
Slight deviation at the extreme right tail -> presence of a few large positive residu
Minor deviation at lower tail -> small negative outliers possible.
'''

# -----------------------------------------------------------------------------
# FINAL EDA SUMMARY
# -----------------------------------------------------------------------------

'''
1.Multiple variables influence car Mileage.

2.Strong predictors identified using correlation.

3.Multicollinearity checked using VIF.

4.Residual diagnostics confirm model assumptions.

5.Model useful for Mileage strategy and feature impact analysis.
'''

#=================================
#Data Preprocessing:- (CARS DATASET)
#================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from feature_engine.outliers import Winsorizer

# -----------------------------------------------------------------------------
# STEP 2: LOAD DATASET
# -----------------------------------------------------------------------------
cars = pd.read_csv("c:/22_Multilinear_Regression/Cars.csv")

print("Initial Shape:", cars.shape)
print(cars.head())

# -----------------------------------------------------------------------------
# STEP 3: BASIC CLEANING
#------------------------------------------------------------------------------
print("\nData Types:\n", cars.dtypes)
print("\nMissing Values Before Treatment:\n", cars.isnull().sum())
'''
Inference:
Dataset contains only numerical variables.
No categorical encoding required.
If missing values exist -> must be handled.
MPG is target variable.
'''

# -----------------------------------------------------------------------------
# STEP 4: MISSING VALUE TREATMENT (Median Imputation)
# -----------------------------------------------------------------------------
for col in cars.columns:
    cars[col].fillna(cars[col].median(), inplace=True)

print("\nMissing Values After Treatment:\n", cars.isnull().sum())
'''
Inference:
Median used because:
Robust to outliers.
Suitable for skewed numerical data.
Prevents distortion of regression coefficients.
'''


# -----------------------------------------------------------------------------
# STEP 5: DUPLICATE REMOVAL
# -----------------------------------------------------------------------------
cars.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:", cars.shape)

'''
Inference:
Removes repeated vehicle records.
Prevents model bias.
Improves generalization capability.
'''

# -----------------------------------------------------------------------------
# STEP 6: OUTLIER DETECTION
# -----------------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=cars, orient='h')
plt.title("Boxplot Before Treatment")
plt.show()
'''
Inference:
HP and SP show extreme values.
VOL and WT may show strong correlation.
Outliers can distort regression coefficients.
'''
#==========================================
#Outlier Tratment(WINSORIZATION)
#==========================================
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=list(cars.columns)      #Convert to list
)

cars = winsor.fit_transform(cars)
cars = winsor.fit_transform(cars)

plt.figure(figsize=(8,5))
sns.boxplot(data=cars, orient='h')
plt.title("Boxplot After Winsorization")
plt.show()

'''
Inference:
Extreme values capped using IQR method.
Reduces impact of abnormal vehicles.
Improves regression stability.
'''
#------------------------------------------------------------------------------
# STEP 8: SKEWNESS CHECK
# -----------------------------------------------------------------------------
print("\nSkewness:\n", cars.skew())

'''
Inference:
• Skewness > 1 -> Strong skew (Log transformation may help).
• Mild skew -> acceptable for regression.
• Helps decide transformation strategy.

HP = 1.06
Slightly above 1
Moderate right skew
Winsorization already reduced extreme values
Transformation optional (Log can help, but not mandatory)
SP = 0.74
Mild skew
No strong need for transformation
VOL, WT
Mild left skew
Not serious
No transformation required
MPG (Target)
Almost symmetric
No treatment required

Final Conclusion
After winsorization:
Skewness is moderate, not severe
No compulsory left/right transformation required
Linear regression is robust to mild skewness
Only transform if:
Residuals violate normality badly
Model fit is poor
'''
# -----------------------------------------------------------------------------
# STEP 9: TRAIN-TEST SPLIT
# -----------------------------------------------------------------------------
X = cars.drop(columns=["MPG"])
y = cars["MPG"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

'''
Inference:
80% data used for training.
20% data used for testing.
Ensures model generalization.
'''

'''
Why Standardization Is NOT Required in OLS
Linear Regression formula:
Y = β₀ + β₁X₁ + β₂X₂ + ...

OLS estimates coefficients using least squares.
It does NOT depend on:
Distance calculation
Gradient scaling sensitivity (like neural networks)
So model works fine without scaling.
'''
#-------------------------------------------------------------------------------
#FINAL DATA PREPROCESSING SUMMARY:
#-------------------------------------------------------------------------------
'''
Dataset validated and cleaned.

Missing values handled using median.

Duplicates removed.

Outliers treated using IQR-based winsorization.

Skewness evaluated.

Data split into training and testing sets.

Dataset is now ready for Multiple Linear Regression modeling.


'''
    
# ======================================================================
# STEP 10: BUILD MULTIPLE LINEAR REGRESSION MODEL
# ======================================================================
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Combine training data into one dataframe for formula API
cars_train = pd.concat([X_train, y_train], axis=1)
cars_test = pd.concat([X_test, y_test], axis=1)

# Initial Model with all predictors
ml1 = smf.ols('MPG ~ HP + VOL + SP + WT', data=cars_train).fit()
print(ml1.summary())

'''
Inference:
1 Model Strength
R-squared = 0.831
Model explains 83.1% of variation in MPG.
This is a strong model fit.

1 Model Strength
R-squared = 0.831
Model explains 83.1% of variation in MPG.
This is a strong model fit.
Indicates predictors collectively explain fuel efficiency well.

Adjusted R-squared = 0.820
Very close to R².
Means no unnecessary variables inflating model.

Model is stable.

2 Overall Model Significance

F-statistic = 72.55
Prob(F-statistic) = 4.23e-22 (< 0.05)
Does the independent variable(s) collectively explain the dependent variab
Or is the relationship happening just by chance?
Model is statistically significant.
At least one predictor significantly affects MPG.

3 Coefficient Interpretation
Variable   Coefficient p-value Interpretation
HP         -0.235      0.000   Significant
VOL        -0.513      0.320   Not significant
SP          0.463      0.007   Significant
WT          0.984      0.536   Not significant

HP (Horsepower)
Highly significant (p < 0.05)
Negative coefficient
Higher horsepower → Lower MPG
Logical: Powerful cars consume more fuel.

SP (Speed)
Significant (p = 0.007)
Positive relationship
Higher speed capability slightly improves MPG
Could indicate aerodynamic efficiency.
VOL (Engine Volume)
Not statistically significant

Likely affected by multicollinearity.

4 Residual Diagnostics

Durbin-Watson Test (Independence of Errors)
The Durbin-Watson statistic checks whether residuals are independent.

DW Value        Meaning
≈ 2             No autocorrelation (Ideal)
< 1.5           Positive autocorrelation
> 2.5           Negative autocorrelation
1.5 - 2.5       Acceptable
Durbin-Watson = 1.721
 Value is close to 2.
 Lies within acceptable range (1.5-2.5).
 No serious autocorrelation problem.
 Residuals are reasonably independent.

Normality Check:Jarque-Bera p-value
Residuals are approximately normally distributed.
Regression assumption satisfied.

5 Multicollinearity Warning
Condition Number = 6530 (Very High )

Indicates:
Strong multicollinearity present.
VOL & WT likely highly correlated.
Coefficients may be unstable.
'''

# ======================================================================
# STEP 11: MULTICOLLINEARITY CHECK (VIF)
# ======================================================================


# Calculating VIF manually
rsq_hp  = smf.ols('HP ~ VOL + SP + WT', data=cars_train).fit().rsquared
vif_hp  = 1/(1-rsq_hp)

rsq_vol = smf.ols('VOL ~ HP + SP + WT', data=cars_train).fit().rsquared
vif_vol = 1/(1-rsq_vol)

rsq_sp  = smf.ols('SP ~ HP + VOL + WT', data=cars_train).fit().rsquared
vif_sp  = 1/(1-rsq_sp)

rsq_wt  = smf.ols('WT ~ HP + VOL + SP', data=cars_train).fit().rsquared
vif_wt  = 1/(1-rsq_wt)

vif_frame = pd.DataFrame({
    'Variable':['HP','VOL','SP','WT'],
    'VIF':[vif_hp, vif_vol, vif_sp, vif_wt]
})

print("\nVIF Values:\n", vif_frame)

'''
# ----------------------------------------------------------------------
# VIF INTERPRETATION & COLUMN REMOVAL DECISION
# ----------------------------------------------------------------------

Variance Inflation Factor (VIF) Interpretation:
Rule of Thumb:
VIF = 1      → No multicollinearity
VIF 1-5      → Moderate (Acceptable)
VIF > 5      → High correlation concern
VIF > 10     → Severe multicollinearity (Problematic)

From the VIF results:
    VOL and WT show very high VIF values.
→ Indicates strong multicollinearity.
→ These two variables are highly correlated with each other.
→ Keeping both will make coefficients unstable.

HP and SP may also show moderately high VIF.
→ Indicates performance-related variables are correlated.

Column Omission Decision:

1 Identify the variable with highest VIF.
2 Remove the variable with:
  - Highest VIF
  - Higher p-value
  - Lower business relevance

In this case:
WT can be removed because:
  - It has very high VIF.
  After removing WT:
• Recalculate VIF.
• Refit the regression model.
• Check if remaining variables have VIF < 10.

Final Goal:
Reduce multicollinearity to improve:
Coefficient stability
Model interpretability
Statistical significance reliability
'''
# ======================================================================
# STEP 12: DROP HIGH VIF VARIABLE (Example: WT)
# ======================================================================

final_ml = smf.ols('MPG ~ HP + VOL + SP', data=cars_train).fit()
print(final_ml.summary())

'''
# ----------------------------------------------------------------------
# OLS REGRESSION RESULTS - INTERPRETATION
# ----------------------------------------------------------------------


1 Model Strength

R-squared = 0.830
Model explains 83.0% of variation in MPG.
This is a strong model fit.
Indicates predictors collectively explain fuel efficiency well.

Adjusted R-squared = 0.821
Very close to R².
No unnecessary variables inflating the model.
Model is stable and reliable.


2 Overall Model Significance

F-statistic = 97.60
Prob(F-statistic) = 4.75e-23 (< 0.05)
purpose of F-statistic
Does the independent variable(s) collectively explain the dependent variab
Or is the relationship happening just by ch


3 Coefficient Interpretation

HP (Horsepower)
 Coefficient = -0.234
 p-value = 0.000 (Highly significant)
 Higher horsepower → Lower MPG.
 Powerful engines consume more fuel.

VOL (Engine Volume)
 Coefficient = -0.195
 p-value = 0.000 (Highly significant)
 Larger engine volume → Lower MPG.
 Bigger engines reduce fuel efficiency.

SP (Speed)
 Coefficient = 0.468
 p-value = 0.006 (Significant)
 Higher speed capability slightly increases MPG.
 May indicate better aerodynamic efficiency.

Intercept

4 Residual Diagnostics
The Durbin-Watson statistic checks whether residuals are independent.
DW Value    Meaning
≈ 2 No autocorrelation (Ideal)
< 1.5       Positive autocorrelation
> 2.5       Negative autocorrelation
1.5-2.5 Acceptable
Durbin-Watson = 1.714

Value is close to 2.
Lies within acceptable range.
No serious autocorrelation problem.
Residuals are reasonably independent.

Normality Check:
Jarque-Bera p-value = 0.625 (> 0.05)

Residuals are approximately normally distributed.

5 Multicollinearity Warning

Condition Number = 6.44e+03 (Very High)

Indicates possible multicollinearity.
Predictors may be correlated.
Coefficients may be sensitive to small data changes.
Interpretation should be done carefully.
'''

# ======================================================================
# STEP 13: ASSUMPTION CHECKING
#=======================================================================
# Predictions
train_pred = final_ml.predict(cars_train)
test_pred  = final_ml.predict(cars_test)

# Residuals
residuals = final_ml.resid

# --- QQ Plot ---
sm.qqplot(residuals)
plt.title("QQ Plot - Residuals")
plt.show()
'''
Interpretation:
 Residuals are approximately normally distributed.
 Minor tail deviations indicate presence of a few mild outliers.
 No severe skewness or heavy-tailed behavior observed.
Conclusion:
 Normality assumption of linear regression is satisfied.
 Model residuals behave well.
The regression model is statistically reliable from a normality perspecti

'''

# --- Residual vs Fitted ---
sns.residplot(x=train_pred, y=y_train, lowess=True)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

'''
Interpretation:
Linearity assumption is mostly satisfied.
 No severe heteroscedasticity detected.
 Minor curvature suggests slight model misspecification.
 Some outliers may influence the model.
'''
# ======================================================================
# STEP 14: MODEL EVALUATION (RMSE)
# ======================================================================

train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
test_rmse  = np.sqrt(np.mean((test_pred - y_test)**2))

print("Train RMSE:", round(train_rmse,4))
print("Test RMSE :", round(test_rmse,4))

'''
Interpretation Guide:

Train RMSE < Test RMSE  → Normal case
Train RMSE ≈ Test RMSE  → Ideal
Train RMSE >> Test RMSE → Underfitting
Train RMSE << Test RMSE → Overfitting
'''

# ======================================================================
# FINAL MODEL SUMMARY
#========================================================================
'''
Data preprocessed successfully.

Multicollinearity checked using VIF.

High VIF variable removed.

Model assumptions verified.

Model evaluated using Train/Test RMSE.
'''
------------------------------------------------------------
BUSINESS IMPACT
------------------------------------------------------------
1️ Strategic Impact:
Enables automobile manufacturers to design vehicles aligned with fuel-efficiency regulations and sustainability goals.
Supports long-term R&D strategy by identifying performance–efficiency trade-offs.
Strengthens competitive positioning in markets where fuel economy is a key purchase driver.

2️ Financial Impact:

Reduction in fuel consumption improves product attractiveness, leading to higher sales.
Optimized engine and vehicle design reduces material and manufacturing costs (weight optimization).
Minimizes regulatory penalties related to emission and fuel economy standards.
Improves ROI on product development by focusing investment on high-impact features.

3️ Operational Impact:

Assists engineering teams in data-driven vehicle design decisions.
Reduces trial-and-error experimentation during prototype development.
Supports simulation-based testing instead of expensive physical testing.
Enables faster product development cycles using predictive modeling.

4️ Regulatory & Environmental Impact:

Helps manufacturers meet government fuel efficiency and emission norms.
Contributes to lower carbon footprint and environmental sustainability.
Aligns with global climate goals and green mobility initiatives.

️⃣ 5 Customer & Market Impact:

Provides customers with more fuel-efficient and cost-effective vehicles.
Improves brand reputation as an environmentally responsible manufacturer.
Enables data-backed marketing claims (e.g., optimized MPG performance).
Helps customers make informed purchasing decisions based on efficiency metrics.

6️ Analytical & Organizational Impact:

Promotes adoption of data-driven decision-making in automotive engineering.
Builds internal analytics capability within product design teams.
Creates foundation for advanced modeling (non-linear regression, ML models).
Enables continuous performance monitoring and improvement.

7️ Long-Term Impact:

Supports transition toward hybrid and electric vehicle optimization strategies.
Provides scalable modeling framework for future vehicle platforms.
Establishes predictive intelligence in product lifecycle management.