#=====================================================================
#Business Understanding:-
#======================================================================
1. Business Problem Statement
People who get into accidents sometimes hire lawyers to get more money
from insurance companies. For the insurance company, lawyers cause 
problems because they lead to bigger payouts, longer case times, and 
extra legal costs.

2. Business Objective
The company wants to use customer details (like age, location, and how 
bad the accident was) to predict who will hire a lawyer. If they can 
guess early, they can step in, settle things quickly, and save money.

3. Motivation
Catch costly cases early.
Send company lawyers only where needed.
Save money by avoiding court fights.
Set better insurance prices.

4. Constraints & Limitations
Some data won’t be useful.
If very few people hire lawyers, the model won’t learn well.
One huge accident can throw off the results.
Data doesn’t show emotions or moods.
Logistic regression can only find simple patterns.

5. Business Success Criteria
The company spots high-risk cases early and settles them before lawyers
get involved.

ML Success Criteria
Patterns are real, not by chance.
Model catches the right people with few false alarms (aim for 70%+ 
accuracy).
Works much better than guessing.
Performs well on new customers too.
#============================================================ 
 DATA UNDERSTANDING:-
#===========================================================
'''
Feature Name    Description                               Type      Business Rele

ATTORNEY        Whether claimant hired attorney (0/1)     Binary    Target variabl

CLMAGE          Age of claimant                           Numeric   Demographic in

LOSS            Claim amount                           continuous   Financial seve

CLMINSUR        Insurance coverage indicator              Binary    Policy-related

CLMSEX          Gender of claimant                        Binary    Demographic be

SEATBELT        Seatbelt usage indicator                  Binary    Injury severit
'''
# =========================================================================
# LOGISTIC REGRESSION - EXPLORATORY DATA ANALYSIS (CLAIMANTS)
# =========================================================================

# -------------------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------------------------------
# STEP 2: LOAD DATA
# -------------------------------------------------------------------------

claimants = pd.read_csv("C:/24_Logistic_Regression/claimants.csv")

# Drop unnecessary column
c1 = claimants.drop('CASENUM', axis=1)

print("First 5 Rows:\n", c1.head())
print("\nData Types:\n", c1.dtypes)
print("\nSummary Statistics:\n", c1.describe())
print("\nMissing Values:\n", c1.isnull().sum())

"""
Inference:
- Check variable types (numeric / categorical).
- Ensure ATTORNEY is binary (0/1).
- Identify missing values for imputation.There are several missing 
  values are there
- LOSS likely has high variance -> strong predictor candidate.
"""


'''
What Are Four Moment Business Decisions?

Mean

Variance

Skewness

Kurtosis

They describe distribution shape of numerical variables.
They are very important in Linear Regression, but their role changes in
Logistic Regression.
In Linear Regression
They are critical because:
Normality assumption required (for residuals)
Homoscedasticity matters
Outliers affect coefficients strongly
So 4 moments directly influence:
Model validity
Hypothesis testing
Confidence intervals

 In Logistic Regression
Logistic regression does NOT require normality.
It assumes:
log (𝑝/1−𝑝)=𝛽0+ 𝛽1+....
So what matters more?
 Linearity in log-odds
 Multicollinearity
 Class balance
 Separation issues
 '''
# -------------------------------------------------------------------------
# STEP 3: TARGET VARIABLE ANALYSIS (VERY IMPORTANT)
# -------------------------------------------------------------------------
print("\nTarget Distribution:\n", c1['ATTORNEY'].value_counts())

sns.countplot(x='ATTORNEY', data=c1)
plt.title("Class Distribution of ATTORNEY")
plt.show()

"""
Inference:

1. The target variable ATTORNEY is nearly balanced.
2. Class 0 (No Attorney) and Class 1 (Attorney) have almost equal frequency.
3. No significant class imbalance problem is observed.
4. Logistic regression can be applied directly without resampling techniques.
5. Accuracy will be a reliable metric since baseline accuracy is ~50%.
6. Model will not be biased toward any dominant class."""
#--------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS (NUMERICAL FEATURES)
# -------------------------------------------------------------------------
c1[['CLMAGE', 'LOSS']].hist(figsize=(10,6))
plt.suptitle("Histogram of Numerical Variables")
plt.show()

"""
Inference:

1 CLMAGE
Distribution appears slightly right-skewed.
Majority of claimants are in the 20-50 age group.
Few older claimants (above 70-80) create a tail.
No extreme abnormal pattern observed.
Implication for Logistic Regression:
CLMAGE looks reasonably distributed.
No immediate transformation required.
Can be used directly in the model.

2 LOSS

Distribution is highly positively skewed (strong right skew).
Most claim amounts are small.
Few very large claims create a long right tail.
Presence of extreme values.
Implication for Logistic Regression:
Strong skewness may distort log-odds relationship.
Consider applying log transformation:
"""

# -------------------------------------------------------------------------
# STEP 5: OUTLIER DETECTION (BOXPLOT):-
#---------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=c1[['CLMAGE', 'LOSS']])
plt.title("Boxplot for Outlier Detection")
plt.show()

"""
Inference:
1 CLMAGE
Moderate spread observed.
Few upper outliers (very high ages).
Majority of ages lie within reasonable range.
No extreme abnormal variation.
Model Impact:
CLMAGE does not show severe outlier problem.
Can be safely used in logistic regression.
Outliers are minimal and manageable.

2.Loss:
    
    very high number of upper outliers.
    string right skewness confirmed.
    few extremely large claim amounts(upto 170+).
    Most values clustered near Lower range.
    Model Impact:
    LOSS has significant extreme values.
   These may strongly influence log-odds.
   Recommended to apply winsorizor 
"""
#=================================================================
#STEP6:BIVARIATE ANALYSIS(TARGETVS NUMERICAL)
#=================================================================

sns.boxplot(x='ATTORNEY', y='LOSS', data=c1)
plt.title("LOSS vs ATTORNEY")
plt.show()

sns.boxplot(x='ATTORNEY', y='CLMAGE', data=c1)
plt.title("CLMAGE vs ATTORNEY")
plt.show()

"""
Inference:

Age does not show a very strong separation between hiring and not 
hiring an attor[ney]
Since distributions overlap heavily, CLMAGE alone may not be a strong
predictor.
It may still contribute when combined with other variables (like LOSS)."""

---------------------------------------------------------------------------
STEP 7: CATEGORICAL vs TARGET ANALYSIS
---------------------------------------------------------------------------
print("\nCLMSEX vs ATTORNEY:\n", pd.crosstab(c1.CLMSEX, c1.ATTORNEY, normalize='index'))
"""
Inference:

For CLMSEX = 0
55.46% -> Did NOT hire attorney
44.53% -> Hired attorney
For CLMSEX = 1
47.43% -> Did NOT hire attorney
52.56% -> Hired attorney
Claimants with CLMSEX = 1 have a higher probability of hiring an attorney (~52.6%).
Claimants with CLMSEX = 0 are slightly less likely to hire (~44.5%).
Difference is moderate, not extreme.
"""

print("\nCLMINSUR vs ATTORNEY:\n", pd.crosstab(c1.CLMINSUR, c1.ATTORNEY, normalize='index'))
"""
For CLMINSUR = 0
63.33% -> Did NOT hire attorney
36.67% -> Hired attorney
For CLMINSUR = 1
49.62% -> Did NOT hire attorney
50.38% -> Hired attorney

Interpretation
Claimants with CLMINSUR = 1 have a higher probability (~50.4%) of hiring an attorney.
Claimants with CLMINSUR = 0 are much less likely (~36.7%) to hire an attorney.
The difference (~14%) is noticeable.

...
"""
print("\nSEATBELT vs ATTORNEY:\n", pd.crosstab(c1.SEATBELT, c1.ATTORNEY, normalize='index'))
""""
Categorical (SEATBELT) vs Target (ATTORNEY)
Observed Proportions
For SEATBELT = 0
50.63% -> Did NOT hire attorney
49.37% -> Hired attorney
For SEATBELT = 1
72.73% -> Did NOT hire attorney
27.27% -> Hired attorney

Interpretation
Claimants not wearing seatbelt (0) show almost equal probability of hiring attorney (~49%)
Claimants wearing seatbelt (1) are much less likely (~27%) to hire attorney.
Difference is substantial (~22%).
This suggests:
SEATBELT is a strong discriminator for attorney hiring."""


#-------------------------------------------------------------------------------
#STEP8:-CORRELATION MATRIX
#-------------------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(c1.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
"""
Inference:
    1.Correlation with Target (ATTORNEY)
    
 Variable    Correlation with ATTORNEY         INTERPRETATION
| -------- | ------------------------- | -------------------------- |
| CLMSEX   | +0.08                     | Very weak positive         |
| CLMINSUR | +0.079                    | Very weak positive         |
| SEATBELT | -0.057                    | Weak negative              |
| CLMAGE   | +0.011                    | Almost no relationship     |
| LOSS     | -0.22                     | Moderate negative          |

2 Multicollinearity Check (Among Predictors)
Look at predictor-to-predictor correlations:
All correlations are very low (close to 0).
No strong correlations (> 0.7 or < -0.7).
Highest among predictors is very small (~0.11).
Inference:
No multicollinearity problem.
VIF values are expected to be low.
Model coefficients will be stable. """

# ---------------------------------------------------------------------------
# STEP 9: MULTICOLLINEARITY CHECK (VIF) - FIXED VERSION
# ---------------------------------------------------------------------------
# 1. Drop target variable to keep only predictors
X = c1.drop(columns=['ATTORNEY'])

# 2. Replace infinite values with NaN safely
X = X.replace([np.inf, -np.inf], np.nan)

# 3. Drop rows containing missing values (All letters must be uppercase 'X')
X = X.dropna()

# 4. Add Constant intercept for the statsmodels VIF calculation
X_const = add_constant(X)

# 5. Compute VIF data frame
vif_data = pd.DataFrame()
vif_data["Feature"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i)
                   for i in range(X_const.shape[1])]

print("\nVariance Inflation Factor:\n", vif_data)
"""
Inference:
- VIF > 10 -> severe multicollinearity]
- Remove variable if VIF high.
- Multicollinearity affects coefficient stability.
| Feature  | VIF   | Interpretation                     |
| -------- | ----- | ---------------------------------- |
| const    | 13.06 | Ignore (intercept not interpreted) |
| CLMSEX   | 1.00  | No multicollinearity               |
| CLMINSUR | 1.00  | No multicollinearity               |
| SEATBELT | 1.02  | No multicollinearity               |
| CLMAGE   | 1.00  | No multicollinearity               |
| LOSS     | 1.02  | No multicollinearity               |
Key Observations
All predictor variables have VIF ≈ 1.
VIF < 5 indicates no multicollinearity problem.
Predictors are almost independent of each other.
Coefficient estimates in logistic regression will be stable.

"""
# ---------------------------------------------------------------------------
# STEP 10: SKEWNESS CHECK
# ---------------------------------------------------------------------------
print("\nSkewness:\n", c1.skew())

"""

Skewness Inference:

ATTORNEY (0.04):
Target variable is nearly symmetric -> confirms balanced class 
distribution

CLMSEX (-0.23):
Slight negative skew -> almost balanced categorical variable -> 
no issue.

CLMINSUR (-2.81):
Highly negatively skewed -> one category dominates -> moderate 
imbalance.

SEATBELT (7.47):
Extremely positively skewed -> strong dominance of one category.

CLMAGE (0.41):
Mild positive skew -> acceptable distribution -> no transformation
 required.

LOSS (7.72):
Extremely positively skewed -> many small claims and few very large 
claims.
Recommended to apply log transformation to stabilize effect in 
logistic regression.

Overall:
Logistic regression does not require normality,
but extreme skewness (especially LOSS) may influence log-odds strongly.
If skewness > 1 -> apply log transformation.
"""
# =========================================================
# Step11:-Check linearity in Logit (FIXED)
# =========================================================
# Logistic regression assumes linearity in log-odds

# 1. FIXED: Changed np.logip to np.log1p
c1['LOSS_Log'] = np.log1p(c1['LOSS'])

y = c1['ATTORNEY']
# 2. FIXED: Column names now match perfectly ('LOSS_Log')
X = c1[['CLMAGE', 'LOSS_Log']]

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN in either X or y
data = pd.concat([X, y], axis=1).dropna()

X_clean = sm.add_constant(data[['CLMAGE', 'LOSS_Log']])
y_clean = data['ATTORNEY']

# Fit model
model_temp = sm.Logit(y_clean, X_clean).fit()

"""
What to Observe:
(A) Converged
Must be True
If False -> model unreliable
Meaning:
MLE optimization successfully found solution.

(B) LLR p-value (Likelihood Ratio Test)

If:
LLR p-value < 0.05 -> Model is significant
LLR p-value > 0.05 -> Model not useful
Meaning:
At least one predictor affects probability.

This is the overall model significance test.

(C) Pseudo R²
Not same as linear regression R².
Interpretation guideline:
0.02 -> weak
0.10-> moderate
0.20+ -> strong(for Logistic)

Meaning:
    How much improvement oer null model.
    
    #Never interpret it like Linear regression R2
    
(D) Log-Likelihood

LL-Null -> model without predictors

Log-Likelihood -> with predictors

If Log-Likelihood improves (less negative) -> model better.

Inference:

Logistic Regression Inference:

Model Summary:

-Model converged successfully -> estimation is reliable.

-LLR p-value = 2.62e-56 (< 0.05) -> Model is statistically significant.

-Pseudo R² = 0.1608 -> Model explains ~16% variation in log-odds.
(Moderate explanatory power for logistic regression.)

Intercept (const = 0.9892, p < 0.001):

-Baseline log-odds of hiring an att[orney without predictors = 0.
                                    
CLMAGE (coef = 0.0089, p = 0.007):
- Positive and statistically significant.
- As age increases, probability of hiring attorney slightly increases.
- Odds ratio ≈ exp(0.0089) ≈ 1.009
  → Each 1-year increase in age increases odds by ~0.9%.

LOSS_log (coef = -1.4231, p < 0.001):
- Strong negative and highly significant predictor.
- As log(LOSS) increases, probability of hiring attorney decreases.
- Odds ratio ≈ exp(-1.4231) ≈ 0.24
  → Higher loss reduces odds of hiring attorney by ~76%.

Overall Interpretation:
- Both CLMAGE and LOSS_Log significantly influence attorney hiring.
- LOSS_log is the strongest predictor (large magnitude and very small p-value).
- Model is statistically strong but with moderate predictive power.
"""

# ---------------------------------------------------------------------------
# FINAL EDA SUMMARY
# ---------------------------------------------------------------------------
"""
1. The Core Business Problem & Goal
The Problem: When accident claimants hire a lawyer, it costs the 
insurance company a lot of money and slows everything down.

The Goal: Predict who will hire a lawyer early on so the company can
 reach out and settle the claim peacefully and cheaply.

2. Data Health Check (The Good News)
Perfect Target Balance: The target column (ATTORNEY) is split almost 
50/50 between people who hired a lawyer and those who didn’t. This 
makes it very easy for the computer to learn without any special tweaks.

No Multicollinearity: The features are independent of each other (all
VIF scores are around 1). This means the columns aren't repeating the 
same information, keeping our model stable.

3. Key Feature Insights (What Actually Matters)
The Claim Amount (LOSS) is the Heavyweight: This column has extreme
right skewness (lots of small claims, few massive ones). The math 
proves that LOSS is the strongest predictor of whether someone hires a
lawyer. To keep this extreme skewness from messing up the model,
applying a log transformation (LOSS-log) is highly recommended.

Seatbelts Matter (SEATBELT): People wearing a seatbelt are much less
likely (27%) to hire a lawyer compared to those who weren't wearing one
(49%). This is a strong, helpful pattern.

Insurance Status Matters (CLMINSUR): Claimants who already have 
insurance are more likely (50.4%) to hire an attorney than uninsured 
ones (36.7%).

Age & Gender are Weak: Age (CLMAGE) and Gender (CLMSEX) show huge 
overlaps in the charts, meaning they only offer minor help in 
predicting a lawyer hire.

4. Final Verdict for Modeling
The initial test model converged successfully and is statistically
 meaningful (p-value is way below 0.05). It explains about 16% of the
behavior right out of the gate.

By cleaning up the extreme values in LOSS using log transformation, 
you are perfectly set up to build a highly stable and reliable Logistic
Regression model!
"""
#========================================================  
DATA PREPROCESSING FOR LOGISTIC REGRESSION
#======================================================== 

#------------------------------------------------------
#STEP1: IMPORT REQUIRED LIBRARIES
#---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer


#------------------------------------------------------------
STEP2: LOAD DATASET
#------------------------------------------------------------    
claimants = pd.read_csv("c:/24_Logistic_Regression/claimants.csv")

print("Initial Shape:", claimants.shape)
print(claimants.head())

# ---------------------------------------------------------------------------
# STEP 3: BASIC CLEANING
# ---------------------------------------------------------------------------
# Drop unnecessary column
c1 = claimants.drop(columns=["CASENUM"])  

#Convert special symbols to NaN if present
c1 = c1.replace(['?', 'NA', 'N/A', 'null', 'NULL', ' '], np.nan)

print("\n Dta Types:\n")
print(c1.dtypes)

print("\nMissing Values Before Treatment:\n")
print(c1.isnull().sum())            
#-------------------------------------------------------------------------------
#STEP4:MISSING VALUE Treatment
#------------------------------------------------------------------------------
# 1. Numerical variable - Median Imputation for Age
c1["CLMAGE"] = c1["CLMAGE"].fillna(c1["CLMAGE"].median())

# 2. FIXED: Categorical variables - Mode Imputation (imputing each column with its OWN mode)
c1["CLMINSUR"] = c1["CLMINSUR"].fillna(c1["CLMINSUR"].mode()[0])
c1["SEATBELT"] = c1["SEATBELT"].fillna(c1["SEATBELT"].mode()[0])

print("\n Missing Values after Treatment:\n")
# 3. FIXED: Changed .null() to .isnull()
print(c1.isnull().sum())

print("\nTotal Missing Values Remaining:", c1.isnull().sum().sum())

'''
Inference:
Median used for CLMAGE ->Robust to outliers
Mode ued for categorical variables-> preserves class distribution.
Dataset now free from missing values.
'''

#-----------------------------------------------------------------------
#Step5:-Duplicate Removal
#-------------------------------------------------------------------------
c1.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:", c1.shape)

'''
Inference:
    Removes repeated claimant records
    prevents bias in probability estimation
    improves model generalization.
'''   

#---------------------------------------------------------------------------------
#STEP6:OUTLIER DETECTION
# ---------------------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=c1[["CLMAGE","LOSS"]])
plt.title("Boxplot Before Treatment")
plt.show()
'''
Inference:
CLMAGE shows mild outliers
LOSSS shows strong right-skew and extreme values
Extreme values may distort Log-odds
'''
#-------------------------------------------------------------------------------------
#STEP7:OUTLIER TREATMENT(OPTIONAL FOR LOGISTIC)
#-------------------------------------------------------------------------------------   

c1 = c1.dropna(subset=["CLMAGE", "LOSS"])

# Now apply Winsorizer
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=["CLMAGE","LOSS"]
)
c1 = winsor.fit_transform(c1)

plt.figure(figsize=(8,5))
sns.boxplot(data=c1[["CLMAGE","LOSS"]])  
plt.title("Boxplot After Winsorization")
plt.show()
'''
Inference:
• Extreme claim values capped.
• Reduces influence of abnormal cases.
• Improves stability of logistic coefficients.
'''  
#---------------------------------------------------------------------------------  
#Business Impact:
#--------------------------------------------------------------------------
1. Stopping Big Cash Leaks Immediately
What the data found: The claim amount (LOSS) is the 1 reason people 
hire a lawyer.

The Business Impact: Instead of waiting for a lawyer to surprise the 
company with a massive bill,the system can instantly flag high-value 
claims the exact day they are submitted.By reaching out to these 
customers with a fair, upfront settlement offer right away,the company
saves millions on inflated court costs.

2. Spotting High-Risk Red Flags (Seatbelts)
What the data found: People who weren’t wearing a seatbelt are almost
twice as likely to hire a lawyer (49% vs 27%).

The Business Impact: The company can set up an automatic alert. If an 
accident report shows "No Seatbelt," that file is immediately 
prioritized. A friendly customer care representative can call them 
right away to settle their medical claims before they get angry and
run to a law firm.

3. Saving Staff Time & Expenses
What the data found: Simple traits like age and gender don't influence
 whether someone hires a lawyer.
The Business Impact: Team managers can stop wasting time profiling
customers by personal background. Instead, the company can deploy its
high-paid defense lawyers only to the truly chaotic, tough cases that
have both high financial losses and safety failures.

4. Smarter Policy Pricing
What the data found: Certain customer behaviors (like seatbelt usage 
and existing coverage) create clear risk patterns.

The Business Impact: The sales team can reward safe drivers (who wear
seatbelts and are unlikely to sue) with cheaper insurance premiums,
while charging a higher, fairer price to profiles that are 
statistically high-risk.

