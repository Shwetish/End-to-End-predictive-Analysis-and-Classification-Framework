
#=============================================================================
# Business Understanding:-
#=============================================================================
1 Business Problem Statement:
  When packages get stuck or take too long to get sorted inside the warehouse, it
  creates a bottleneck. Logistics managers need to know exactly how these warehouse
  delays drag down the final delivery time to the customers doorstep.

2 Business Objective:
- Find the Link: See exactly how much a delay in sorting impacts the final delivery 
  time.
- Build a Predictor: Create a smart tool that forecasts exactly when a package will
  arrive based on how long it take to sort.
- Smart Planning: Use this data to schedule drivers better, plan routes, and know 
  when to add more staff to the warehouse floor.
  
3 Motivation:
- It stops packages from arriving late to customers.
- It helps us build realistic, reliable driving schedules.
- It keeps us from breaking our promises to clients.

4 Constraints:
- The Wildcard Elements: Traffic, bad weather, and driver speeds change constantly and
  randomly.
- Limited Data: We might not have a massive amount of historical data to work with.
- Unpredictable Patterns: The relationship between sorting and delivery is not always
  a perfectly straight line.
- Freak Delays: One massive warehouse breakdown or a major storm can mess up our math
  and make our regular formulas inaccurate.

5 Success Criteria:
  Business Success:
- We can accurately guess delivery times, and managers get clear, 
  useful insights to run smoother daily operations.

6.ML Success:
- Our AI model makes very small guessing errors, proves a clear
  connection between sorting and delivery, and works reliably on brand-new, 
  real-world data.

-----------------------------------------------------------------------
DATA UNDERSTANDING:-
-----------------------------------------------------------------------
'''
Feature Name        Description Type              Business Relevance
Sorting_Time        Warehouse sorting time (mins) Numeric Key predictor
Delivery_Time       Total delivery time (mins)    Numeric Target variable
'''

# =======================================================================
# EXPLORATORY DATA ANALYSIS (9 DISTINCT STEPS)
# =======================================================================

#-------------------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES FOR ANALYSIS
#-----------------------------------------------------------------------
--
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------------------
# STEP 2: LOAD THE DATASET
#-----------------------------------------------------------------------
# Loading the delivery dataset from the same directory
deliv = pd.read_csv("C:/21_Linear_Regression_Colory/delivery_time.csv")

# Standardizing column names for easy analysis
deliv.columns = ["delivery_time", "sorting_time"]

print("First 5 Rows:\n", deliv.head())

#-----------------------------------------------------------------------
# STEP 3: BASIC EDA
#-----------------------------------------------------------------------
print("\nData Types:\n", deliv.dtypes)
print("\nSummary Statistics:\n", deliv.describe())

# -----------------------------------------------------------------------
# BUSINESS MOMENT DECISIONS
# -----------------------------------------------------------------------

# Mean
print("\nMean:\n", deliv.mean())
'''
Inference:
• On average, packages spend 6.19 units of time being sorted in the warehouse.
• On average, the total delivery turnaround time takes 16.79 units of time.
• Sorting takes up 6.19 units of time, and total delivery takes 16.79 units.
• This means warehouse sorting eats up over one-third (37%) of the total timeline.
• Because it is such a big chunk of the process, any delay in sorting will instantly cause a 
  late delivery.
'''

# Variance
print("\nVariance:\n", deliv.var())
'''
Inference:
• High variance in sorting time -> warehouse operations are inconsistent day-to-day.
• High variance in delivery time -> customers experience unpredictable arrival times.
• Large spread makes predicting final delivery times trickier.
'''

# Standard Deviation
print("\nStandard Deviation:\n", deliv.std())
'''
Inference:
• Shows the typical deviation or "gap" from the average times.
• Low values -> highly predictable and stable logistics system.
• High values -> unstable operations with fluctuating transit times.
'''

# Skewness
print("\nSkewness:\n", deliv.skew())
'''
Inference:
• Skew ≈ 0 -> Balanced, symmetrical operational times.
• Positive skew -> A few severe operational delays or massive backlogs are pulling the data to 
  the right.
'''

# Kurtosis
print("\nKurtosis:\n", deliv.kurtosis())
'''
Inference:
• High kurtosis -> Frequent "freak events" or extreme peak-season delays.
• Low kurtosis -> Most daily delivery cycles look highly similar and uniform.
'''

# Correlation
print("\nCorrelation Matrix:\n", np.corrcoef(deliv.sorting_time, deliv.delivery_time))
'''
Inference:
• The correlation between sorting time and delivery time is 0.83.
• This means there is a strong, positive link between them: as sorting time goes up,
  delivery time goes up too.
• Because this link is strong, we can confidently use sorting time to predict final delivery
  times.
'''
# -----------------------------------------------------------------------
# STEP 4: UNIVARIATE ANALYSIS
# -----------------------------------------------------------------------

# Histogram - Delivery Time
plt.figure(figsize=(6,4))
plt.hist(deliv.delivery_time, color='skyblue', edgecolor='black')
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (Hours/Days)")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
• Most deliveries fall within a standard, expected time window.
• A right-skewed tail means a few orders experience major delays in the field.
• These long delays can impact customer satisfaction scores.
'''

# Histogram - Sorting Time
plt.figure(figsize=(6,4))
plt.hist(deliv.sorting_time, color='salmon', edgecolor='black')
plt.title("Sorting Time Distribution")
plt.xlabel("Sorting Time (Minutes/Hours)")
plt.ylabel("Frequency")
plt.show()

'''
Inference:
• Shows how quickly the warehouse staff clears packages.
• Any unusual peaks on the higher side show specific shifts where the warehouse gets heavily
  clogged.
'''

#-----------------------------------------------------------------------
# STEP 4: BOX PLOT (OUTLIER DETECTION)
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=deliv[['delivery_time', 'sorting_time']], orient='h', palette='Set2')
plt.title("Boxplot of Delivery & Sorting Features")
plt.show()

'''
Inference:
• No outliers detected: All data points sit inside the whiskers, meaning there are no extreme
  freak delays in sorting or delivery.
• Steady performance: Most deliveries take between 13.5 and 19.5 units of time (the green box), showing a highly consistent routine.
• Sorting limit: The warehouse maximum sorting time caps out right at 10 units of time, 
  keeping operational bottlenecks contained.
'''
#-----------------------------------------------------------------------
# STEP 5: BIVARIATE ANALYSIS (SCATTER PLOT)
#-----------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(x='sorting_time', y='delivery_time', data=deliv, color='purple')
plt.title("Sorting Time vs Delivery Time")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

'''
Inference:
• Upward trend indicates: More time spent sorting = later delivery times.
• If data points form a tight upward line, warehouse speed heavily dictates delivery success.
• If points are scattered wildly, outside factors (like traffic or distance) matter more than 
  sorting.
'''

#-----------------------------------------------------------------------
# STEP 6: CORRELATION HEATMAP
#-----------------------------------------------------------------------
plt.figure(figsize=(5,4))
sns.heatmap(deliv.corr(), annot=True, cmap='YlGnBu')
plt.title("Logistics Correlation Heatmap")
plt.show()

'''
Inference:
• Closer the number is to 1.0, the stronger the link between sorting and final delivery.
• High correlation -> Warehouse automation and staffing will directly fix late deliveries.
• High correlation justifies using a Simple Linear Regression model to predict arrival times.
'''

#-----------------------------------------------------------------------
# PDF & CDF ANALYSIS
#-----------------------------------------------------------------------
for col in ['delivery_time', 'sorting_time']:
    plt.figure(figsize=(12,4))
    
    # PDF
    plt.subplot(1,2,1)
    sns.kdeplot(deliv[col], fill=True, color='teal')
    plt.title(f'PDF of {col}')
    
    # CDF
    plt.subplot(1,2,2)
    sorted_vals = np.sort(deliv[col])
    y = np.arange(len(sorted_vals)) / len(sorted_vals)
    plt.plot(sorted_vals, y, color='darkorange', linewidth=2)
    plt.title(f'CDF of {col}')
    
    plt.show()
    
'''
1. PDF (Probability Density Function) Analysis:
• Shows the "sweet spot" where your sorting and delivery times happen most often.
• Helps spot if your logistics process has multiple peaks (e.g., fast morning shifts
  vs. slow night shifts).

2. CDF (Cumulative Distribution Function) Analysis:
• A steep initial curve means your team handles the vast majority of orders very 
 quickly.
'''

# -----------------------------------------------------------------------
# FINAL EDA SUMMARY
# -----------------------------------------------------------------------
"""
FINAL SUMMARY:-
• Confirms whether warehouse processing speed acts as the main domino affecting 
  delivery times.
• Quantifies delivery predictability, helping managers set accurate delivery
  expectations for clients.
• Clear patterns validate that Linear Regression can reliably predict final delivery 
  schedules.
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
deliv = pd.read_csv("C:/21_Linear_Regression_Colory/delivery_time.csv")
deliv.columns = ["delivery_time", "sorting_time"]

print("Initial Shape:", deliv.shape)
print(deliv.head())

'''
Inference:
• We successfully loaded the logistics data.
• The dataset has two core features: our target 'delivery_time' and our predictor
 'sorting_time'.
'''

#----------------------------------------------------------------------
# STEP 3: BASIC CLEANING & MISSING VALUE TREATMENT
#----------------------------------------------------------------------
print("\nMissing Values Before Treatment:\n", deliv.isnull().sum())

# Robust handling: Median Imputation
for col in deliv.columns:
    deliv[col] = deliv[col].fillna(deliv[col].median())

print("\nMissing Values After Treatment:\n", deliv.isnull().sum())

'''
Inference:-
-Data is perfectly clean: No missing values were found in either column (0 blank spots).
-Code safety net: The median filling step did not change anything because the data is
 already complete, but it remains in the code to handle any missing values in future 
 data files.
'''

#-----------------------------------------------------------------------
# STEP 4: DUPLICATE REMOVAL 
#-----------------------------------------------------------------------
deliv.drop_duplicates(inplace=True)
print("\nShape After Removing Duplicates:", deliv.shape)

'''
Inference:
• Identical repeated rows can trick the model into over-learning specific patterns.
• We cleared out all duplicate entries to ensure the data is unique and fair.
'''

#-----------------------------------------------------------------------
# STEP 5: OUTLIER DETECTION & TREATMENT (WINSORIZATION)
#-----------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=deliv, orient='h', palette='Set2')
plt.title("Boxplot Before Treatment")
plt.show()

# Applying Winsorizer using IQR method
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=['delivery_time', 'sorting_time']
)
deliv[['delivery_time', 'sorting_time']] = winsor.fit_transform(deliv[['delivery_time', 'sorting_time']])
'''
Inference
-No outliers found: The boxplot shows there are no random, extreme "freak delays" in 
either column since all data points sit safely inside the whiskers.

-The Winsorizer step didn't need to change anything because the 
data is already clean, but it acts as a great automatic safety filter for future
data files.
'''
#-----------------------------------------------------------------------
# STEP 6: SKEWNESS CHECK
#-----------------------------------------------------------------------
print("\nSkewness:\n", deliv.skew())

'''
Inference:
• Skewness tells us if our delivery records lean heavily toward one side.
• A score close to 0 means the daily operational workload is stable and 
  symmetrically distributed.
'''

#------------------------------------------------------------------------
# MODEL DEVELOPMENT - COMPARING MULTIPLE MODEL TRANSFORMATIONS
# -----------------------------------------------------------------------

# =======================================================================
# MODEL 1: SIMPLE LINEAR REGRESSION (SLR)
# =======================================================================
model1 = smf.ols('delivery_time ~ sorting_time', data=deliv).fit()
pred1 = model1.predict(deliv)
rmse1 = np.sqrt(np.mean((deliv.delivery_time - pred1)**2))

print("\n--- MODEL 1: SLR SUMMARY ---")
print("SLR RMSE:", rmse1)
print(model1.summary())

'''
Inference
Good accuracy: The model has an R-squared of 0.682, which means sorting time 
explains about 68.2% of the changes in final delivery time.

Small guessing error: On average, the model's delivery time guesses are off by only
 about 2.79 units of time (RMSE).

Strong positive link: The sorting_time score (1.6490) tells us that for every 1 
extra minute a package spends being sorted, the final delivery time increases by 
about 1.65 minutes.

Highly reliable: The P-value (P>|t|) is 0.000 (well below 0.05), proving that 
warehouse sorting time is a highly trustworthy factor for predicting delivery scores
'''.

# =======================================================================
# MODEL 2: LOG MODEL (log X)
# =======================================================================
model2 = smf.ols('delivery_time ~ np.log(sorting_time)', data=deliv).fit()
pred2 = model2.predict(deliv)
rmse2 = np.sqrt(np.mean((deliv.delivery_time - pred2)**2))

print("\n--- MODEL 2: LOG-X SUMMARY ---")
print("Log-X RMSE:", rmse2)
print(model2.summary())

'''
Inference:
InferenceGood Accuracy ($R\text{-squared} = 0.695$): About 69.5% of the changes in
delivery times are explained by this model.
Highly Significant ($P\text{-value} = 0.000$): The log-transformed sorting time is a 
reliable predictor of delivery time; this relationship did not happen by random chance
Average Guessing Error ($\text{RMSE} = 2.73$): On average, the model's delivery time
predictions are off by about 2.73 units of time (e.g., minutes or hours).
Diminishing Impact Trend: Because it uses a log scale, it shows that initial warehouse
sorting delays push delivery times up quickly, but extra delays past a certain point 
have a flattening impact on the final delivery timeline.
'''
# =======================================================================
# MODEL 3: EXPONENTIAL MODEL (log Y)
# =======================================================================
model3 = smf.ols('np.log(delivery_time) ~ sorting_time', data=deliv).fit()
pred3 = model3.predict(deliv)
pred3_back = np.exp(pred3) 
rmse3 = np.sqrt(np.mean((deliv.delivery_time - pred3_back)**2))

print("\n--- MODEL 3: EXPONENTIAL SUMMARY ---")
print("Exponential RMSE:", rmse3)
print(model3.summary())

'''
Inference:-
Strong Accuracy ($R\text{-squared} = 0.711$): This model explains 71.1% of the
 variance in delivery times, which is slightly higher than the Log-X model.
 Highly Significant ($P\text{-value} = 0.000$): There is a definitive relationship
 between sorting time and the logarithmic delivery time. 
 It is a highly dependable predictor.Average Guessing Error ($\text{RMSE} = 2.94$):
 When converted back to original units, the model's predictions are off by an average
 of 2.94 units of time. 
 Even though the $R\text{-squared}$ improved, the absolute guessing error is slightly
 higher than Model 2.Snowballing Delay Trend: Because it uses a log scale on the 
 delivery time ($Y$), this model captures an exponential pattern—meaning each
 additional minute of warehouse sorting delay causes delivery times to pile up at an 
 accelerating rate.
 '''

# =======================================================================
# MODEL 4: POLYNOMIAL MODEL (Quadratic)
# =======================================================================
model4 = smf.ols('delivery_time ~ sorting_time + I(sorting_time**2)', data=deliv).fit()
pred4 = model4.predict(deliv)
rmse4 = np.sqrt(np.mean((deliv.delivery_time - pred4)**2))

print("\n--- MODEL 4: POLYNOMIAL SUMMARY ---")
print("Polynomial RMSE:", rmse4)
print(model4.summary())

'''
Inference:-Good Overall Accuracy ($R\text{-squared} = 0.693$): 
This model explains 69.3% of the changes in delivery times.
Low Guessing Error($\text{RMSE} = 2.74$): On average, the model's delivery time 
predictions are off by about 2.74 units of time.
Unnecessary Complexity ($P\text{-values}$ are high): 
Look at the individual features: sorting_time ($0.070$) and sorting_time  2 ($0.429$)
both have high $P\text{-values}$ (greater than $0.05$). 
This means adding the squared curve term does not provide a statistically meaningful
benefit over a simpler straight line.
Curved Trend: The negative coefficient ($-0.0932$) for the squared term means the
prediction line creates a slight downward-bending curve, attempting to show that 
delivery times level off slightly at extremely high sorting times.
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

best_model_row = results.sort_values("RMSE").iloc[0]
best_model_name = best_model_row["Model"]
print(f"\nBest Performing Model Configuration: {best_model_name}")

'''
Inference:
• The Winner is Log-X: The Log-X model is chosen as the best configuration because 
it has the absolute lowest guessing error ($\text{RMSE} = 2.733$).
Understanding Accuracy ($\text{R-squared}$): While the Exponential model has a 
slightly higher accuracy score ($71.1\%$), it actually makes larger individual
guessing errors in real time ($\text{RMSE} = 2.940$). 
Log-X strikes the best balance by capturing a solid $69.5\%$ of the data patterns 
with the lowest overall mistakes.
The Final summary: Since warehouse sorting and final delivery do not follow a 
perfectly straight line, transforming the sorting data using a logarithm 
($\text{Log-X}$) gives managers the most reliable, real-world
predictions.
'''

# =======================================================================
# STEP 8: TRAIN-TEST VALIDATION
# =======================================================================
train, test = train_test_split(deliv, test_size=0.3, random_state=42)

final_model = smf.ols('delivery_time ~ sorting_time', data=train).fit()

train_pred = final_model.predict(train)
test_pred = final_model.predict(test)

train_rmse = np.sqrt(np.mean((train.delivery_time - train_pred)**2))
test_rmse = np.sqrt(np.mean((test.delivery_time - test_pred)**2))

print("\n--- TRAIN-TEST CROSS VALIDATION ---")
print("Train Split RMSE:", train_rmse)
print("Test Split RMSE :", test_rmse)

'''
Inference:-
 Stable Model: The training error ($2.77$) and testing error ($3.35$) are close.
 This proves the model is reliable and not just memorizing the data.
 Ready for Use: The model is ready to predict arrival times for brand-new deliveries
 with a small real-world guessing error of about $3.35$ units of time.
'''

# =======================================================================
# STEP 9: VISUALIZE FINAL MODEL FITS
# =======================================================================
plt.figure(figsize=(10, 6))
plt.scatter(deliv.sorting_time, deliv.delivery_time, color='blue', label='Actual Deliveries', alpha=0.7)
plt.plot(deliv.sorting_time, pred1, color='red', linewidth=2, label=f'SLR Line (R²={model1.rsquared:.3f})')
plt.title("Final Model Fit: Sorting Time vs Delivery Time")
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.legend()
plt.show()

'''
Inference:
• This plot shows our mathematical formula in action. 
• The red prediction line cuts right through our scatter points, proving that 
  tracking sorting speed gives us a highly accurate window into final delivery times.
'''
#==========================================================================
#10. Business Impact of the Solution
#==========================================================================
 1️.Fix Warehouse Bottlenecks
 Impact: Sorting packages takes up over 37% of the total
 delivery time.
 Action: Managers can invest in faster tools like smart scanners or 
 conveyor belts to speed up sorting and quicken overall deliveries.
 2.Smarter Staff Scheduling
 Impact: Warehouse workloads change wildly and randomly 
 day-to-day.
 Action: Instead of keeping the same staff size, managers can use a 
 data-driven calendar to add extra workers only on busy days before packages pile up
 3.Better Route Planning
 Impact: There is a strong link ($0.83$) between sorting speed
 and final delivery success.
 Action: If sorting runs slow, dispatchers can use the 
 models math to change driver routes and schedules early to prevent late arrivals.
 4️. Precise Delivery Promises
 Impact: Testing multiple models ensures we pick the
 formula with the lowest guessing error.
 Action: Support teams can give customers 
 exact, trustworthy delivery times, keeping clients happy and avoiding late penalties.
 
 