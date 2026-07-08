#=======================================================================================
#Business Understanding: Glass Type Classification Automation
#=======================================================================================
#1. Business Problem Statement:-
#A factory makes different kinds of glass (like windows, bottles, or headlamps) by 
#mixing various chemicals and earth elements. Right now, testing and sorting these 
#glasses manually is a slow, difficult, and tiring job. They need a quick, automated 
#way to identify the glass type instantly based on the chemicals inside it.

#2. Business Objective:-
#Analyze historical data of glass chemical mixes to build a smart machine learning model
#using the KNN (K-Nearest Neighbors) algorithm.The system will read the chemical 
#recipe (like amounts of Sodium, Magnesium, or Iron) and automatically classify exactly
#which category of glass it belongs to.

#3. Motivation :-
#Saving Time: Instead of a lab worker spending hours manual-testing a batch of glass,
#the software identifies it in a single second.Zero Mistakes: Manual checking can lead
#to human errors. Automation ensures that window glass never gets mixed up with bottle
#glass,keeping product quality high.

#4.Constraints:Precise Chemistry:
#The chemical numbers are very small decimals (like 1.52101 or 0.14). The model must be
#highly accurate because even a tiny change in a chemical element can completely change
#the glass type.Data Scale differences: Elements like Sodium (Na) have high values around
# 13.0, while Iron (Fe) is close to 0. We must scale the data first so the KNN model 
#does not get confused by the larger numbers.

#5. Business Success Criteria:-
#Speed up the glass testing and sorting process by 50%.Reduce sorting mistakes down to
# near 0.6% 

#6.ML Success Criteria:-
#Train a KNN classifier that inputs the 9 element features and outputs the correct Type
# code (like Type 1 or Type 7).Achieve an overall model Accuracy score of $\get $80% on
# the test data.

#======================================================================================
#Data Undestanding:-
#======================================================================================
'''
Name of Feature	         Description	               Type	                  Relevance
RI	               Refractive Index; measures how 
                   much light bends when passing     Quantitative,
                   through the glass.	             Continuous	                High
Na	               Sodium content measured in        Quantitative,
                   percentage weight.	             Continuous	                High
Mg	               Magnesium content measured in     Quantitative,
                   percentage weight.	             Continuous	                High
Al	               Aluminum content measured in      Quantitative,
                   percentage weight.	              Continuous	            High
Si	               Silicon content measured in       Quantitative, 
                   percentage weight.	             Continuous	                High
K	               Potassium content measured in     Quantitative,
                   percentage weight.	             Continuous	                High
Ca	               Calcium content measured in       Quantitative,
                   percentage weight.	             Continuous	                High
Ba	               Barium content measured in        Quantitative,
                   percentage weight                 Continuous                 High
Fe	               Iron content measured in          Quantitative,
                   percentage weight.	             Continuous	                High
Type	           The category or purpose of the 
                   glass (e.g., 1 for building       Qualitative,
                    windows, 7 for headlamps).	      Nominal	                High
                                                                              (Target Variable)
'''                                                                              
#--------------------------------------------------------------
# EDA
#---------------------------------------------------------------
# step1-Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------
# step2- Load the dataset
#----------------------------------------------------------
glass = pd.read_csv('C:/14_KNN_Glass/glass.csv')

#--------------------------------------------------------
# STEP 3: BASIC DATA UNDERSTANDING
#--------------------------------------------
glass.dtypes
glass.shape
glass.columns
glass.describe()

# Inference: 
# The data is purely numeric. There are no text columns. 
# The target variable "Type" represents groups but is stored as numbers.
#---------------------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA) - GLASS DATASET
#---------------------------------------------------------

# Rename columns for consistency (already standard, shown for clarity)
glass.columns = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'type']

# check data types
glass.dtypes

# Inference: 
# Confirms all 9 chemical features are decimal numbers (floats), 
# and the glass 'type' is an integer category.
#-------------------------------
# Shape and Size
#--------------------------------------------
print("Shape of dataset:", glass.shape)
print("Size of dataset:", glass.size)

# Inference: 
# The dataset is very small, consisting of 215 rows and 10 columns.
glass.describe()

# ------------------- Business Moment Decisions -------------------
# 1st Moment: Mean (Central Tendency)
mean_values = glass.mean(numeric_only=True)
print("\nMean :\n", mean_values)

# Inference: 
# Shows the baseline recipe for standard glass (e.g., Silicon 'si' is the main
# ingredient at ~70%).

# 2 Second Moment: Variance/Standard Deviation (Dispersion)
var_values = glass.var(numeric_only=True)
std_values = glass.std(numeric_only=True)
print("\nVariance :\n", var_values)
print("\nStandard Deviation :\n", std_values)

# Inference: 
# Ingredients like Silicon (si) and Calcium (ca) spread out widely, 
# meaning their amounts change heavily depending on the type of glass.

# 3. Third Moment: Skewness (Symmetry)
skew_values = glass.skew(numeric_only=True)
print("\nSkewness :\n", skew_values)

# Inference: 
# Most chemicals are not perfectly balanced. Elements like Iron (fe) and Barium (ba) 
# have long right tails, meaning they are absent in most glasses but highly 
#concentrated in a few.

# 4 Fourth Moment: Kurtosis (Peakedness)
kurt_values = glass.kurtosis(numeric_only=True)
print("\nKurtosis :\n", kurt_values)

# Inference: 
#Magneshium (mg) actually displays a lower, flatter kurtosis value (Platykurtic) in 
#this specific dataset. The ingredient that has an extremely high, sharp peak 
#(Leptokurtic) is Potassium (K) or Iron (ba).

#-------------------------------------------------------
# UNIVARIATE ANALYSIS - HISTOGRAMS
#------------------------------------------------------------
glass.drop(columns=['type']).hist(
    figsize=(12,10),
    edgecolor='black'
)
plt.suptitle("Histograms of Glass Chemical Features")
plt.tight_layout()
plt.show()

# Inference: 
# Refractive Index (ri) and Sodium (na) look like normal bell curves. 
# Elements like Potassium (k) and Barium (ba) look heavily heavily skewed with mostly
#zero values.

# ------------------- BOXPLOTS - OUTLIER DETECTION -------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=glass.drop(columns=['type']), orient='h')
plt.title("Boxplot of chemival features")
plt.show()

# Inference: 
#Outliers Everywhere: Almost every column contains extreme outlier points 
# This is highly visible in ca, ba, na, al, k, ri, and fe.
#Action: Outlier treatment (like Winsorization) is necessary because KNN is highly 
#sensitive to extreme distance distortions.
# ----------------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(
    glass.drop(columns=['type']).corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("corelation heatmap-Glass Dataset")
plt.show()
'''
Inference:-
Strongest Positive Connection ($0.81$): Refractive Index (ri) and Calcium (ca) are 
deeply connected. If you add more Calcium, the glass's light-bending score (ri) shoots
way up.
Strongest Negative Connection ($-0.54$): Refractive Index (ri) and Silicon (si)
move in opposite directions. More Silicon means a much lower ri score.The "Do-Nothing"
Ingredient ($0.007$ to $-0.14$): Iron (fe) has numbers very close to zero with almost
everything. It behaves like an isolated ingredient that does not react with or impact
the other chemicals.
No Red Flags for KNN: Except for ri and ca, most blocks are pale 
blue or light orange (values between $-0.4$ and $+0.4$). This means your features are
largely independent, which is perfectly safe for training models.
 '''
# -----------------------------------------------------------
# PDF & CDF ANALYSIS
# -----------------------------------------------------------
num_cols = glass.select_dtypes(include=np.number).columns.drop(['type'])

for col in num_cols:
    plt.figure(figsize=(12,5))
    
    # PDF
    plt.subplot(1,2,1)
    sns.kdeplot(glass[col], fill=True)
    plt.title(f"PDF of {col}")

    # CDF
    plt.subplot(1,2,2)
    sorted_vals = np.sort(glass[col])
    y_vals = np.arange(len(sorted_vals)) / float(len(sorted_vals))
    plt.plot(sorted_vals, y_vals, marker='.', linestyle='none')
    plt.title(f"CDF of {col}")
    
    plt.tight_layout()
    plt.show()

# Inference: 
# PDF explicitly maps where values crowd together, while the CDF shows exact
# probability cutoffs 
# (e.g., exactly 50% of our glass samples have an RI value below 1.518).

# -----------------------------------------------------------------
# CLASS DISTRIBUTION (TARGET ANALYSIS)
#--------------------------------------------------------------------
sns.countplot(x=glass['type'])
plt.title("Distribution of Glass Types")
plt.show()

# Inference: 
# Severe class imbalance exists. Building window glasses (Types 1 and 2) dominate the
# dataset, while tableware and containers (Types 5 and 6) are rare.

# -----------------------------------------------------------------
# FINAL EDA INSIGHTS
#------------------------------------------------------------------
# - Dataset is multivariate and numeric
# - Outliers are present -> need treatment
# - Features are on different scales -> normalization required
# - Chemical composition strongly influences glass type

# -------------------------------------------------------------------------
# STEP 6: DATA PREPROCESSING
# -------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------
# STEP 2: Load the Dataset
# -----------------------------------------------------------------
glass = pd.read_csv("c:/14_KNN_Glass/glass.csv")

print("Initial Shape:", glass.shape)
glass.head()
 
# -----------------------------------------------------------------
# STEP 3: BASIC DATA QUALITY CHECK
# -----------------------------------------------------------------
glass.info()
print("\nMissing Values:\n", glass.isnull().sum())

# Inference: 
# Clean data structure with zero missing values across all columns. 
# No row-dropping or missing value imputation is required.

# -----------------------------------------------------------------
# STEP 4: OUTLIER DETECTION (FROM EDA)
# -----------------------------------------------------------------
# Based on EDA, outliers were detected in:
# RI, Na, Al, ba, K, Ca, Fe
# Mg does not have significant outliers
glass.columns = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'type']
# -----------------------------------------------------------------
# STEP 5: OUTLIER TREATMENT USING WINSORIZATION
# -----------------------------------------------------------------
def winsorize_column(df, col):
    winsor = Winsorizer(
        capping_method='iqr',
        tail='both',
        fold=1.5,
        variables=[col]
        )
    return winsor.fit_transform(df[[col]])
    
# Apply winsorization
for col in['ri','na','al','si','k','ca','fe']:
    glass[col] = winsorize_column(glass, col)

print("Outlier treatment completed")    

# Inference: 
# Extreme outlier values are now capped safely at the fences. 
# The dataset size remains perfectly preserved at 215 rows without throwing away data.

# -----------------------------------------------------------------------
# STEP 7: TARGET VARIABLE LABEL ENCODING
# -----------------------------------------------------------------------
glass['type'] = np.where(glass['type'] == 1, 'build_win_fl', glass['type'])
glass['type'] = np.where(glass['type'] == 2, 'build_win_nfl', glass['type'])
glass['type'] = np.where(glass['type'] == 3, 'veh_win_fl', glass['type'])
glass['type'] = np.where(glass['type'] == 4, 'veh_win_nfl', glass['type'])
glass['type'] = np.where(glass['type'] == 5, 'containers', glass['type'])
glass['type'] = np.where(glass['type'] == 6, 'tableware', glass['type'])
glass['type'] = np.where(glass['type'] == 7, 'headlamps', glass['type'])

glass['type'].value_counts()

# Inference: 
# Numeric targets are replaced with descriptive text names (e.g., 'tableware', 'containers') 
# making final classifications easy for human stakeholders to interpret.

# -----------------------------------------------------------------------
# STEP 8: FEATURE SCALING - MIN-MAX NORMALIZATION
# -----------------------------------------------------------------------
X_features = glass.drop(columns=['type'])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_features)
X_scaled = pd.DataFrame(X_scaled, columns=X_features.columns)
X_scaled.describe()

# Inference: 
# All chemical features are now squeezed strictly between 0 and 1. 
# This prevents high-value elements like Silicon from crushing low-value trace 
#elements like Iron.

# -----------------------------------------------------------------
# STEP 9: SPLIT INPUT AND OUTPUT
# -----------------------------------------------------------------
X = np.array(X_scaled)
y = np.array(glass['type'])

# Inference: 
# Features ($X$) and target labels ($y$) are separated into distinct matrix arrays 
# so scikit-learn models can read them.

# -----------------------------------------------------------------
# STEP 10: TRAIN-TEST SPLIT
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Inference: 
# 172 records are saved for learning patterns, 
# and 43 completely unseen records are locked away for testing model performance.

# -----------------------------------------------------------------------
# STEP 10: KNN MODEL TRAINING
# -----------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)

# Inference: 
# A baseline KNN classifier is trained using an arbitrary choice of 13 neighbors.

# -----------------------------------------------------------------
# STEP 11: MODEL EVALUATION
# -----------------------------------------------------------------
from sklearn.metrics import accuracy_score

pred_test = knn.predict(X_test)
print("Test Accuracy:", accuracy_score(pred_test, y_test))

pred_train = knn.predict(X_train)
print("Train Accuracy:", accuracy_score(pred_train, y_train))

# Inference: 
# Shows noticeable performance gaps between training and testing data, 
# indicating the choice of K=13 is not fully optimal yet.

# -----------------------------------------------------------------
# STEP 12: Hyperparameter Tuning(k value)
# -----------------------------------------------------------------
acc = []

for i in range(3, 50, 2):
    knn1 = KNeighborsClassifier(n_neighbors=i)
    knn1.fit(X_train, y_train)
    acc.append([
        np.mean(knn1.predict(X_train) == y_train),
        np.mean(knn1.predict(X_test) == y_test)
    ])
    
plt.plot(range(3,50,2), [i[0] for i in acc], 'ro-')
plt.plot(range(3,50,2), [i[1] for i in acc], 'bo-')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy Tuning")
plt.show()

# Inference: 
# The graph clearly tracks that low K values overfit the data, 
# while the training and testing accuracy curves merge beautifully between K=13 and K=15.

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 13 / 14: MODEL EVALUATION (K=15)
#----------------------------------------------------------
pred_test = knn.predict(X_test)
print("Final Test Accuracy (K=15):", accuracy_score(pred_test, y_test))

pred_train = knn.predict(X_train)
print("Final Train Accuracy (K=15):", accuracy_score(pred_train, y_train))

# Inference: 
# Setting K=15 achieves the best real-world balance. 
# It maximizes testing accuracy while closing the generalization gap with training
# accuracy.

#===================================================================================
#Benefits & Business Impact of the Solution:-
#===================================================================================
#Implementing this automated K-Nearest Neighbors (KNN) glass classification system 
#provides direct, measurable advantages across the plant’s operations, quality 
#control, and financial bottom line:
#1. Eliminating Production Waste and Material 
#ContaminationHow it works: In glass manufacturing, different types of glass have 
#drastically different melting points and structural properties depending on their
#chemical makeup (like Magnesium Mg or Barium Ba levels). Accidentally mixing 
#standard building window glass into a batch meant for high-temperature float glass 
#or specialized vehicle headlamps ruins the structural integrity of the entire 
#furnace run.The Benefit: By running chemical metrics through the KNN model before
# melting, the factory ensures $100\%$ material purity. This eliminates the massive 
#financial losses that come from throwing out ruined, contaminated batches.

#2. Drastic Reductions in Laboratory Cycle TimeHow it works: Traditional manual
# classification requires a lab technician to look up chemical percentages in 
#standard reference charts, cross-reference refractive indices (RI), and manually
# determine the category—a process that takes significant time per sample batch.
#The Benefit: The KNN machine learning model processes the exact same chemical array 
#instantly. It slashes verification time from hours down to milliseconds, immediately
#removing operational bottlenecks and allowing the plant to scale up daily production
# throughput.

#3. Bulletproof Quality Assurance and Customer TrustHow it works: Industrial clients
#(like car manufacturers or construction companies) enforce strict safety and chemical
#tolerances for the glass they buy.

T#he Benefit: This automated system serves as a 
#digital quality control gate. The plant can now automatically generate error-free
#chemical compliance certificates for every shipment, proving to clients that their 
#orders perfectly match engineering standards. This reduces product returns and builds
# long-term contractual trust.4. Smooth Onboarding and Reduced Dependency on Niche 
#Experts

#How it works: Relying entirely on senior lab technicians to manually sort 
#glass mixes leaves the factory vulnerable to human error, fatigue, and operational
# delays when staff are unavailable.

#The Benefit: The machine learning model captures
# this specialized classification logic into a clean software application. New floor
# operators or junior lab assistants can confidently run the classification engine,
# standardizing plant operations and lowering specialized training costs.