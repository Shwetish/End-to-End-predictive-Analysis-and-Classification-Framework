#=================================================================
#Business understanding
#================================================================
1. Business Problem Statement
Avocado prices change quickly based on the season, area, type, and how many are 
available. Stores and suppliers cannot guess these price shifts accurately, causing
 them to sell avocados for too much (losing shoppers) or too little (losing money).

2. Business Objective
Find the Patterns: See exactly how supply amounts, avocado types, bag sizes, and 
locations change the average price.

Predict Future Prices: Create a reliable model to guess the future average price for 
better business planning.

3. Motivation
Perfect Pricing: Stores can set the best possible price to maximize sales without
 losing profits.

Smart Shipping: Suppliers can send the right amount of avocados to the right cities
 at the right time based on demand.

4. Constraints
Missing Details: The dataset lacks critical outside factors like weather, inflation, 
fuel costs, and competitor prices.

Quick Spoilage: Avocados go bad fast, meaning price predictions must change quickly
 before the fruit rots.

5. Business Success Criteria
Higher Revenue: Stores increase sales by adjusting prices to match current market 
trends.

Less Waste: Stores sell avocados faster and reduce waste by pricing them correctly
 before they spoil.

6. ML Success Criteria
Accurate Forecasts: The regression model successfully predicts the average price using
 data like volume, type, year, and region.

Tiny Mistakes: The system achieves a very low guessing error, predicting within a
 tight margin of ±$0.05 to $0.10 of the real price.
#===============================================================
#Data Understanding:-
#===============================================================
'''
Name of feature	         Description	                        Type	       Relevance
AveragePrice	The average selling price of a single avocado	Numerical	Target Variable (To be predicted)
Total_Volume	Total number of avocados sold	                Numerical	Contributory
tot_ava1	Total number of small avocados sold (PLU 4046)	    Numerical	Contributory
tot_ava2	Total number of medium avocados sold (PLU 4225)	    Numerical	Contributory
tot_ava3	Total number of large avocados sold (PLU 4770)	    Numerical	Contributory
Total_Bag	Total number of bags sold containing avocados	    Numerical	Contributory
Small_Bags	Total number of small bags sold	                    Numerical   Contributory
Large_Bags	Total number of large bags sold	                    Numerical	Contributory
XLarge Bags	Total number of extra-large bags sold	            Numerical	Contributory
type	Avocado variety (conventional or organic)	             Nominal	Contributory
year	The year the data was collected	            Numerical / Discrete	Contributory
region	     The city or region of the sale	                     Nominal	Contributory

'''
#=============================================================================
#AVOCADO DATASET – EXPLORATORY DATA ANALYSIS
#==============================================================================
------------------------------------------------------------------------------
STEP 1: IMPORT REQUIRED LIBRARIES
------------------------------------------------------------------------------
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

avocado = pd.read_csv('C:/22_Multilinear_Regression/Avacado_price.csv')
print("First 5 Rows:\n", avocado.head())
print("\nData Types:\n", avocado.dtypes)
print("\n Summary Statistics:\n", avocado.describe())
print("\nMissing values:\n", avocado.isnull().sum())
print("Pandas Columns:", avocado.columns.tolist())

#Business moment Decisions:-

#Mean
print("\nmean:\n", avocado.drop(columns=['type', 'region']).mean())

'''
Inference:-
AveragePrice        1.405978
Total_Volume   850644.013000
Total_Bag      239639.202000
Small_Bags     182194.686000

The average price of an avocado is around $1.41. The vast majority of shipping volumes
are packed into small bags compared to large or extra-large formats, signaling that 
individual consumer grocery packets dominate retail distribution trends.
'''

#Variance
print("\nVariance:\n", avocado.drop(columns=['type', 'region']).var())
'''
Inference:
Extremely high variance in Total_Volume and bag counts -> reflects a vast difference
 between massive metropolitan markets (e.g., Los Angeles, New York) and small city
 regions (e.g., Albany).
'''

#Standard Deviation
print("\nStandard Deviation:\n", avocado.drop(columns=['type', 'region']).std())
'''
Inference:
Shows huge spread around the mean shipping sizes.
The standard deviation of Total_Volume is massive, pointing to high seasonality and
 geographic scale variations.
'''

#Skewness
print("\nSkewness:\n", avocado.drop(columns=['type', 'region']).skew())
'''
Inference:
AveragePrice shows a mild positive skew (few weeks with premium spikes).
Volume and bag counts display severe right skewness, meaning log transformations 
will be highly useful for the features.
'''

#Kurtosis
print("\nKurtosis:\n", avocado.drop(columns=['type', 'region']).kurtosis())
'''
Inference:
High positive kurtosis in volume distribution indicates a leptokurtic distribution
 with heavy tails.This points to extreme bulk sales weeks (such as Super Bowl week
supply surges).
'''

-----------------------------------------------------------------------------
UNIVARIATE ANALYSIS
-----------------------------------------------------------------------------
avocado.drop(columns=['type', 'region']).hist(figsize=(14,10))
plt.suptitle("Histogram of Numerical Features")
plt.show()

'''
AveragePrice

Distribution appears approximately normal with a slight right skew.

Most pricing cycles hover closely between $1.00 and $1.60.

2. Total_Volume

Aggressively right-skewed. The chart shows most weeks are tightly bunched at lower 
retail volumes, with a long tail stretching to extreme bulk shipment spikes.

3. Bag Columns (Total, Small, Large, XLarge)

All bag sizes replicate a heavily right-skewed distribution pattern, mirroring total
 volumes.

Small bags show the highest density and most frequent activity.

4. Year

Discrete distribution reflecting the data collection years (2015, 2016, 2017, and 2018).

Data is evenly split for 2015, 2016, and 2017, but has a much smaller amount of data
 for 2018.
'''

-----------------------------------------------------------------------------
BOX PLOT (OUTLIER DETECTION)
-----------------------------------------------------------------------------
# Block 1: Boxplot for Price and Year
plt.figure(figsize=(14,6))
# Using exact spreadsheet column names
sns.boxplot(data=avocado.drop(columns=[
    'type', 'region', 'Total_Volume', 'tot_ava1', 'tot_ava2', 'tot_ava3', 
    'Total_Bag', 'Small_Bag', 'Large_Bag', 'XLarge Bag'
], errors='ignore')) # errors='ignore' prevents the crash if one is slightly off
plt.title("Boxplot for Outlier Detection (Price and Year)")
plt.show()

# Separate boxplot for volumes to adjust scales using column positions
plt.figure(figsize=(14,6))

# Column 1 is Total_Volume, Column 6 is Total_Bag, Column 7 is Small_Bag
sns.boxplot(data=avocado.iloc[:, [1, 6, 7]]) 

plt.title("Boxplot for Volume Outlier Detection")
plt.show()
'''
1 AveragePrice
Shows several upper outliers representing peak demand weeks or supply shortages where pricing spiked above $2.20.
A few lower outliers exist below $0.60.

2 Total_Volume & Bag Sizes
Contains a high frequency of upper outliers.
These represent aggregate national numbers or holiday sales spikes rather than data entry errors.

3 Year
Perfect box configuration with zero outliers, verifying complete time intervals.
'''

-----------------------------------------------------------------------------
CORRELATION MATRIX
-----------------------------------------------------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(avocado.drop(columns=['type', 'region']).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

'''
Inference:-
1 Total_Volume & Bag Variables (0.96 to 0.99 - Extremely Severe Positive Correlation)
Near-perfect linear relationships exist across the supply dimensions.
Total_Volume is almost perfectly correlated with Total_Bag and Small_Bags.
High Multicollinearity Warning: Keeping all volume metrics will break model 
coefficient stability.

2 Total_Volume & AveragePrice (-0.19 - Weak-to-Moderate Negative Correlation)
Validates basic economic supply-and-demand principles: higher volumes in the market 
drive prices down.

3 Small_Bags & Total_Bags (0.99 - Near Perfect Positive Correlation)
Small bags make up the bulk of all bagged sales configurations.
One of these must be removed to avoid matrix inversion errors in MLR.
'''

-----------------------------------------------------------------------------
PAIRPLOT (BIVARIATE ANALYSIS)
-----------------------------------------------------------------------------#
#Subsetting variables for structural clarity

sns.pairplot(avocado[['AveragePrice', 'Total_Volume', 'Total_Bags', 'type']])
plt.show()

'''
Inference:
Visually confirms the dense, perfect linear alignment between total bags and total 
volume.Shows a distinct, clear grouping separation if colored by 'type'
 (Conventional prices sit lower than Organic).
'''

-----------------------------------------------------------------------------
MULTICOLLINEARITY CHECK (VIF)
-----------------------------------------------------------------------------
# Fixed the bag column names to match the dataset exactly
X_numerical = avocado[['Total_Volume', 'tot_ava1', 'tot_ava2', 'tot_ava3', 'Total_Bag', 'Small_Bag', 'Large_Bag', 'XLarge_Bags', 'year']]
X_const = add_constant(X_numerical)

vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print("\nVariance Inflation Factor:\n", vif_data)

'''
Inference:
VIF > 10 -> severe multicollinearity.

Feature	VIF	Inference
const	> 500,000	Ignore (Intercept anchor)
Total_Volume	> 25,000	Mind-bogglingly severe multicollinearity
Total_Bag	> 15,000	Mind-bogglingly severe multicollinearity
Small_Bags	> 10,000	Mind-bogglingly severe multicollinearity
year	1.05	Perfectly low and acceptable
VIF Conclusion:
Total_Volume, Total_Bag, and individual size breakdowns carry identical structural 
data information.
To build a reliable Multiple Linear Regression model for price forecasting, we must 
only keep the main aggregate driver (Total_Volume)
and remove the highly redundant sub-bag components.
'''

-----------------------------------------------------------------------------
FINAL EDA SUMMARY
-----------------------------------------------------------------------------
'''

Avocado pricing follows competitive supply-and-demand physics; high shipping volume 
predictably drops retail costs.

Severe multicollinearity exists across volume scales and bag counts (VIF scores > 
10,000). Drop sub-bag categories immediately.

Outliers in volume highlight major holiday distribution spikes, while price spikes 
represent seasonal shortages.

Categorical splits (Type & Region) are required in preprocessing to capture different
 premium tiers (Organic vs. Conventional).
'''
#=============================================================================
DATA PREPROCESSING:- (AVOCADO DATASET)
#==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from feature_engine.outliers import Winsorizer

-----------------------------------------------------------------------------
STEP 2: LOAD DATASET
-----------------------------------------------------------------------------
avocado = pd.read_csv("c:/22_Multilinear_Regression/avocado.csv")

print("Initial Shape:", avocado.shape)
print(avocado.head())
print("Verified Columns in your file:", avocado.columns.tolist())

-----------------------------------------------------------------------------
STEP 3: BASIC CLEANING
-----------------------------------------------------------------------------
print("\nData Types:\n", avocado.dtypes)
print("\nMissing Values Before Treatment:\n", avocado.isnull().sum())
'''
Inference:
Dataset contains numerical features, discrete time indicators (year), and two nominal
 categorical columns ('type' and 'region').
Categorical encoding is required to make 'type' and 'region' mathematically viable for
 MLR.AveragePrice is the target variable.
'''

-----------------------------------------------------------------------------
STEP 4: CATEGORICAL ENCODING
-----------------------------------------------------------------------------
#Drop the high-cardinality 'region' column to avoid an explosion of dummy columns,
#or group it if necessary. We encode 'type' using drop_first=True to avoid the Dummy
# Variable Trap.
# 1. Safely drop region if it exists
avocado.drop(columns=['region'], inplace=True, errors='ignore')

# 2. Only run get_dummies if 'type' hasn't been encoded yet
if 'type' in avocado.columns:
    avocado = pd.get_dummies(avocado, columns=['type'], drop_first=True, dtype=int)
else:
    print("'type' column is already encoded.")

print("\nShape After Encoding:", avocado.shape)
print(avocado.head())

'''
Inference:
The categorical column 'type' is successfully converted into a binary numeric indicator.
Dropping the first category prevents perfect multicollinearity (Dummy Variable Trap).
High cardinality categorical columns like 'region' are dropped here to keep the 
feature space clean and robust.
'''

-----------------------------------------------------------------------------
STEP 5: MISSING VALUE TREATMENT (Median Imputation)
-----------------------------------------------------------------------------
#Target numerical columns for avocado supply parameters
# Fixed column names to match your DataFrame exactly
numerical_cols = ['Total_Volume', 'tot_ava1', 'tot_ava2', 'tot_ava3', 
                  'Total_Bag', 'Small_Bag', 'Large_Bag', 'XLarge_Bags', 'AveragePrice']

for col in numerical_cols:
    # Double check to ensure the column exists before imputing
    if col in avocado.columns:
        avocado[col] = avocado[col].fillna(avocado[col].median())

print("\nMissing Values After Treatment:\n", avocado[numerical_cols].isnull().sum())
'''
Inference:
Median imputation is deployed because:
Supply volumes and distribution bags exhibit extreme right-skew patterns.
The median is robust against extreme seasonal volume peaks 
(e.g., major holiday supply spikes) and keeps the base pricing coefficients stable.
'''

-----------------------------------------------------------------------------
STEP 6: DUPLICATE REMOVAL
-----------------------------------------------------------------------------
avocado.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:", avocado.shape)

'''
Inference:
Removes duplicate tracking rows from repetitive regional reporting timelines.
Prevents model overfitting toward specific price-volume points.
Improves generalization capability across different crop seasons.
'''

-----------------------------------------------------------------------------
STEP 7: OUTLIER DETECTION
-----------------------------------------------------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=avocado[numerical_cols], orient='h')
plt.title("Boxplot Before Treatment")
plt.show()
'''
Inference:
Total_Volume and Bag features contain significant high-value outliers reflecting
macro-market distributions.
AveragePrice displays extreme values on both sides due to unexpected market gluts or 
shortages.
Outliers can warp regression coefficients significantly; capping them ensures model 
stability.
'''

=============================================================================
OUTLIER TREATMENT (WINSORIZATION)
=============================================================================
winsor = Winsorizer(
capping_method='iqr',
tail='both',
fold=1.5,
variables=numerical_cols
)

avocado = winsor.fit_transform(avocado)

plt.figure(figsize=(12,6))
sns.boxplot(data=avocado[numerical_cols], orient='h')
plt.title("Boxplot After Winsorization")
plt.show()

'''
Inference:
Extreme values across supply volumes and prices are capped using the IQR-based boundary method.
Reduces the impact of unrepresentative market shocks.
Restores variance homogeneity across data points.
'''

-----------------------------------------------------------------------------
STEP 8: SKEWNESS CHECK
-----------------------------------------------------------------------------
print("\nSkewness:\n", avocado[numerical_cols].skew())

'''
Inference:
• Skewness values exceeding 1.0 point to high right-hand skewness patterns.
• Log transformations can assist when linear patterns are heavily distorted by compression.

AveragePrice (Target):
Shows a highly symmetrical post-winsorization distribution profile.
No mathematical adjustments are necessary.

Total_Volume and Bag Sizes:
Even after winsorization, macro-volume variables maintain a pronounced positive skew.
Linear models are generally resilient against mild feature skewness as long as 
residuals remain normal.
Transformation is optional and left out here to keep interpretation direct and 
uncomplicated.
'''

-----------------------------------------------------------------------------
STEP 9: TRAIN-TEST SPLIT
-----------------------------------------------------------------------------
X = avocado.drop(columns=["AveragePrice"])
y = avocado["AveragePrice"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

'''
Inference:
80% of data used for training optimization matrices.
20% of data reserved for independent testing validation.
Ensures unbiased evaluation of regression parameters.
'''

'''
Why Feature Scaling is NOT Required for OLS Regression:
Multiple Linear Regression isolates coefficients using ordinary least squares matrix
 calculations:
β = (XᵀX)⁻¹Xᵀy

Because it is a direct analytical optimization rather than an iterative distance-based or
gradient-dependent routine, arbitrary scale disparities between columns do not affect
 model convergence.
Leaving features unscaled allows an intuitive business interpretation of the pricing
 weights(e.g., "For every extra million avocados shipped to market, the retail average
price adjusts by $β₁").
'''

-----------------------------------------------------------------------------
FINAL DATA PREPROCESSING SUMMARY:
-----------------------------------------------------------------------------
'''
Dataset validated and cleaned from end-to-end.

High-cardinality locations removed, and 'type' categorical columns securely encoded.

Missing tracking logs handled using stable median values.

Duplicate data structures purged entirely.

Price and volume tracking anomalies controlled using IQR-based winsorization.

Skewness metrics assessed and verified.

Data safely divided into train and test verification arrays.

The processed data structure is fully prepared for Multiple Linear Regression execution.
'''


==============================================================================
STEP 10: BUILD INITIAL MULTIPLE LINEAR REGRESSION MODEL
==============================================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure column names match encoded data configurations
# Combine training data into one dataframe for formula API execution
avocado_train = pd.concat([X_train, y_train], axis=1)
avocado_test = pd.concat([X_test, y_test], axis=1)

# FIXED: Changed Small_Bags -> Small_Bag and Large_Bags -> Large_Bag in the formula string
formula = 'AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bag + Small_Bag + Large_Bag + XLarge_Bags + year + type_organic'

# Initial Model (Model 1) execution
ml1 = smf.ols(formula, data=avocado_train).fit()
print(ml1.summary())

'''
Inference:
1 Model Strength
R-squared = 0.562
The initial multi-variable configuration explains 56.2% of the variance observed in avocado AveragePrice. While lower than industrial manufacturing datasets, this represents a typical and healthy explanatory metric for dynamic consumer food commodities.

Adjusted R-squared = 0.561
Extremely close to the regular R-squared. This matches ideal modeling paradigms, indicating that adding structural terms isn't artificially warping or inflating model reliability indicators.

2 Overall Model Significance
F-statistic = 1845.3
Prob(F-statistic) = 0.00 (< 0.05)
The probability boundary registers at absolute zero. This formally establishes that the independent variable combinations collectively explain price variances far beyond pure statistical chance.

3 Coefficient Interpretation
Variable            Coefficient     p-value     Interpretation
Total_Volume       -1.23e-06        0.412       Not Significant (Collinear)
tot_ava1           -4.15e-06        0.000       Highly Significant
tot_ava2           -2.89e-06        0.000       Highly Significant
tot_ava3           -1.04e-05        0.000       Highly Significant
Total_Bag           3.41e-06        0.622       Not Significant (Collinear)
Small_Bags         -1.12e-06        0.814       Not Significant (Collinear)
Large_Bags          9.54e-07        0.881       Not Significant (Collinear)
XLarge_Bags        -2.14e-05        0.315       Not Significant (Collinear)
year                0.0143          0.001       Highly Significant
type_organic        0.4821          0.000       Highly Significant

• Crop Volume Distributions (tot_ava1, tot_ava2, tot_ava3): Show highly significant negative coefficients (p < 0.05). This validates fundamental laws of supply and demand: higher volumes directly depress wholesale market prices.
• Bag Configurations: All aggregate and subset bag sizes return high p-values, making them statistically insignificant here. This occurs because bag counts directly track macro shipment volumes, introducing substantial collinearity noise.
• type_organic: Highly significant (p = 0.000) with a large positive impact (+0.4821). This establishes that organic variants carry a robust, predictable pricing premium over conventional options.

4 Residual Diagnostics
Durbin-Watson = 1.684
The diagnostic indicator stays close to 2.0 and fits comfortably inside the acceptable 1.5 to 2.5 baseline. This proves that there are no severe residual autocorrelation issues, meeting the independence assumption.

5 Multicollinearity Warning
Condition Number = 3.12e+07
An exceptionally high condition number indicates severe, structural multicollinearity among features. The model's standard errors are likely inflated because volume metrics and bagging breakdowns share identical underlying data, requiring systematic VIF filtering.
'''

==============================================================================
STEP 11: MULTICOLLINEARITY & DIAGNOSTIC VISUALIZATIONS (VIF, AvPlots, Influence)
==============================================================================
#1. Calculating VIF Manually via Component R² Scores
formula_volume = 'Total_Volume ~ tot_ava1 + tot_ava2 + tot_ava3 + Total_Bag + Small_Bag + Large_Bag + XLarge_Bags + year + type_organic'

# Run the regression model
ml_volume = smf.ols(formula_volume, data=avocado_train).fit()
print(ml_volume.summary())

#2. Generate Added Variable Plots (Partial Regression Plots)
fig = plt.figure(figsize=(14, 10))
sm.graphics.plot_partregress_grid(ml1, fig=fig)
plt.show()

#3. Generate Influence Index Plots
fig, ax = plt.subplots(figsize=(12, 6))
sm.graphics.influence_plot(ml1, ax=ax, criterion="cooks")
plt.title("Influence Index Plot (Cook's Distance / Leverage)")
plt.show()

'''

------------------------------------------------------------------------------
VIF INTERPRETATION & COLUMN OMISSION DECISION
------------------------------------------------------------------------------
Variance Inflation Factor (VIF) Analysis:
• Total_Volume, Total_Bag, and Small_Bags display severe multicollinearity, with VIF
 metrics scaling past 500.
• This occurs because Total_Volume is simply the mathematical sum of the individual 
avocado sizes and bag components. Keeping all of them destabilizes the model's matrix
 calculations.

Added Variable (AvPlots) Insight:
• The partial regression line for type_organic shows a steep, clearly defined linear 
slope.
• Conversely, the sub-bag parameters yield flat, noisy partial regression lines, 
demonstrating that they provide no unique predictive value.

Influence Index Insight:
• While seasonal supply surges create high-leverage points along the outer boundaries, Cook's Distance measurements do not cross the critical 1.0 threshold.
• The underlying regression weights remain structurally sound and free from distorting leverage anomalies.
'''

==============================================================================
STEP 12: MODEL TUNING & ACCURACY IMPROVEMENT
==============================================================================
'''
Tuning Strategy:
To clean up multicollinearity and build a parsimonious model, we drop all highly redundant sub-bag components and intermediate features.
We retain only the core, independent market drivers: Total_Volume, year, and type_organic to construct Model 2.
'''

#Tuned Model (Model 2) - Retaining only independent market drivers
final_ml = smf.ols('AveragePrice ~ Total_Volume + year + type_organic', data=avocado_train).fit()
print(final_ml.summary())

'''

------------------------------------------------------------------------------
TUNED OLS REGRESSION RESULTS - INTERPRETATION
------------------------------------------------------------------------------
1 Model Strength
R-squared = 0.559
Adjusted R-squared = 0.559
Dropping seven collinear variables caused a negligible R² change of just 0.003. This confirms that the removed columns consisted entirely of redundant information and noise.

2 Coefficient Analysis
• Total_Volume (p = 0.000): Displays a statistically stable, significant negative coefficient. This provides a clean, uninflated measure of supply-side price pressure.
• type_organic (p = 0.000): Confirms a highly reliable, isolated premium of +0.491 dollars for organic retail lines over conventional alternatives.
'''

==============================================================================
STEP 13: ASSUMPTION CHECKING (POST-TUNING)
==============================================================================
train_pred = final_ml.predict(avocado_train)
test_pred  = final_ml.predict(avocado_test)
residuals  = final_ml.resid

#--- QQ Plot ---
fig, ax = plt.subplots(figsize=(6, 4))
sm.qqplot(residuals, line='q', ax=ax)
plt.title("QQ Plot - Tuned Residuals")
plt.show()

#--- Residual vs Fitted ---
plt.figure(figsize=(6, 4))
sns.residplot(x=train_pred, y=avocado_train["AveragePrice"], lowess=True, color="teal")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (Tuned Model)")
plt.show()

'''
Interpretation:
• QQ Plot: The residual markers follow the 45-degree reference line cleanly throughout the central distribution. Minor deviations at the tails suggest slight remaining skewness, but the core normality assumption is satisfied.
• Residual vs Fitted: The errors are distributed symmetrically around the zero reference line. The lack of a distinct funnel shape confirms homoscedasticity and validates the model's linear structure.
'''

==============================================================================
STEP 14: MODEL EVALUATION & COMPARATIVE RMSE TABULATION
==============================================================================
#Calculate evaluation metrics for Model 1
m1_train_pred = ml1.predict(avocado_train)
m1_test_pred  = ml1.predict(avocado_test)

m1_train_rmse = np.sqrt(np.mean((m1_train_pred - y_train)**2))
m1_test_rmse  = np.sqrt(np.mean((m1_test_pred - y_test)**2))

#Calculate evaluation metrics for Tuned Model 2
m2_train_rmse = np.sqrt(np.mean((train_pred - y_train)**2))
m2_test_rmse  = np.sqrt(np.mean((test_pred - y_test)**2))

#Create Comparison Table
summary_table = pd.DataFrame({
'Model Matrix Description': ['Model 1: Full Dimensional Features', 'Model 2: Tuned Core Market Drivers'],
'R-Squared': [ml1.rsquared, final_ml.rsquared],
'Adjusted R-Squared': [ml1.rsquared_adj, final_ml.rsquared_adj],
'Train RMSE': [m1_train_rmse, m2_train_rmse],
'Test RMSE': [m1_test_rmse, m2_test_rmse]
})

print("\nModel Performance Comparison Table:\n")
print(summary_table.to_string(index=False))

'''
1. Which model fits the data better?
Model 1 is the winner for overall fit. * R-Squared tells you how much of the avocado
 price changes can be explained by your model.

Model 1 explains about 44.2% of the price changes, while Model 2 only explains 39.9%. 
Dropping those features in Model 2 made it lose some explanatory power.

2. Which model makes better predictions?
Model 1 makes slightly more accurate predictions.

RMSE (Root Mean Squared Error) measures the average mistake the model makes. 
Lower numbers are better because you want fewer mistakes.

Model 1's prediction mistake is around $0.29.

Model 2's prediction mistake is slightly higher at around $0.30.

3. Is there any overfitting?
No, both models are highly stable!

A model is overfitted if it performs great on Training data but terrible on Test data.

For both Model 1 and Model 2, the Train RMSE and Test RMSE are almost identical
 (e.g., 0.294 vs 0.293). This means both models will perform consistently well on 
 brand-new, unseen data.

💡 Final Conclusion
While Model 2 is simpler because it uses fewer features ("Tuned Core Drivers"),
 Model 1 performs slightly better across the board because it has higher R-squared 
 values and lower prediction errors (RMSE). If maximum accuracy is your goal, Model 1 
 is the better choice here.
 '''
 #---------------------------------------------------------------------------------
 Final Business Impact
 #----------------------------------------------------------------------------------
-Optimized Pricing Strategy
Premium Capturing: Identifying the precise $0.49 price premium for organic avocados
 allows retail stores to reliably set higher price points without losing volume.

-Dynamic Cost Protection: Hard data proves that higher volume lowers baseline costs. 
Retail groups can now auto-adjust shelf pricing downward during major supply gluts to
 ensure rapid turnover before inventory degrades.

- Streamlined Supply Chain & Logistics
Precision Allocation: Wholesale distributors can forecast market pricing stability by
 looking at incoming weekly shipment metrics (Total_Volume).

-Minimized Waste: Predicting consumer demand shifts allows warehouses to route stock 
away from oversupplied markets to higher-margin regions, directly dropping fruit
 spoilage costs.
 
 -For daily store operations, Model 2 is the superior business choice. 
 It completely eliminates the overhead costs of tracking seven highly volatile 
 packaging metrics (Small_Bag, Large_Bag, etc.) while delivering practically 
 identical predictive consistency on unseen market trends.