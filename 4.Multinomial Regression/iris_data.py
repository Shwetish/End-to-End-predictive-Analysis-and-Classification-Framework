'''
Correlation with "Playing Football" Analogy

In your sports analogy:

Features = hours playing, watching, practicing (inputs).

Target = chosen sport (Cricket, Football, Tennis).

In the Iris dataset:

Features = flower measurements (inputs).

Target = chosen species (Setosa, Versicolor, Virginia).

So, just Like:

If a student spends many hours on football → higher probability of Football.

If petal length & width are large → higher probability of Virginia.
'''
#-------------------------------------------------------
#1.Import Libraries
#-------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#-------------------------------------------------------
#2.Load Dataset
#---------------------------------------------------------
iris = load_iris()

#Features(X) = Flower Measurements
#Analogy: Hours spent on different activities (Football,Cricket,Tennis practice/watch
x = iris.data


#Target (y) =flower species(setosa,vesicolor, Vigginica
#Anology: which sport is chosen(Football,Cricket,Tennis))
y = iris.target
#---------------------------------------------------
# 2.Split into Train & Test
#-----------------------------------------------------
#Train = learning phase (students practicing sports)
#Test = exam phase(checking if model correctly predicts the sport)
# Recreate X and y from the same cleaned dataset
# Now split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
#----------------------------------------------
#3.Train Multinomial Logistic Regression
#----------------------------------------------
#Model learns "Softmax Equation"
#Analogy :Model learns how hours spent on football /cricket/tennis->probability of
model = LogisticRegression(solver='lbfgs', max_iter=500)
#In scikit-learns Logistic Regression , the parameter solver='lbfgs'
#Specifies that the model will use the L-BFGS optimization algorithm to estimate 
#the regression coefficients(weights).
'''
lbfgs = Limited -memory BFGS - > a smart algorithm that finds the 
best coefficients faster than plain gradient descent.
It is memory-efficient and works well hen we have many features.
It is recommended for multinomial regeression.
saga, sag - good for very large datasets.

'''
model.fit(x_train, y_train)
#-----------------------------------------------------------
#4.Make Predictions:-
#-----------------------------------------------------------

#Model predicts probabilities for each class(species/sports)
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

#------------------------------------------------------------
#5.Evaluate Performance
#-----------------------------------------------------------
#Print precision, recall, F1-Score for each class
#Anology :How well does the model identify "football players" vs

print("Classification Report:\n", classification_report(y_test, y_pred,
                                                        target_names=iris.target_names))

#Class-wise accuracy(per flower type)
print("\nClass-wise Accuracy:")
for i, class_name in enumerate(iris.target_names):
    class_indices = (y_test == i) #where the true class is i
    correct = np.sum(y_pred[class_indices] == y_test[class_indices])
    total = np.sum(class_indices)
    acc = correct / total
    print(f"{class_name}: {acc:.2f}")

'''
The reason you're getting accuracy = 1.0 for all classes is
that your test set happened to be perfectly predicted by
the logistic regression model. This is possible with the Iris dataset,
since it's small, clean, and very well-separated.

But in general, you won't always get perfect accuracy –
it depends on the random split (random_state) and the model performance.

Let me explain clearly:

Why Accuracy = 1 for ALL Classes?

The Iris dataset is Linearly separable,
especially Setosa (always predicted correctly).

Logistic regression with Lbfgs solver converges very well.
--
TypeError                                 Traceback (most recent call last)
Cell In[251], line 1
----> 1 model = LogisticRegression(multi_class='multinomial', solver='lbfgs',
      2                            max_iter=500)

TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
With the 70-30 split,
the test set was easy enough that the model
got everything right.
'''
#------------------------------------------------------------------------------------
#step 6.Example predictions with Explanation
#------------------------------------------------------------------------------------
# 1. Make sure the model generates the predictions first
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

# 2. Now run your print loop
print("\nSample Predictions with Probabilities:")
for i in range(5):  # first 5 test samples
    # Fixed: Changed .iloc[i] to [i] since y_test is a NumPy array
    true_class = iris.target_names[y_test[i]]  
    predicted_class = iris.target_names[y_pred[i]]
    probs = y_proba[i]
    
    print(f"Sample {i+1}: True = {true_class}, Predicted = {predicted_class}, Probabilities = {probs.round(2)}")
    
    print(f" Model says: {predicted_class} is the right choice "
          f"(highest probability {max(probs)*100:.1f}%).\n")
###########################################################################
# Q.2
# =========================================================================
# MULTINOMIAL LOGISTIC REGRESSION ON WINE DATASET
# Complete End-to-End Script with Inline Explanation
# =========================================================================

# Import required libraries
import pandas as pd             # For structured data handling (not heavily used
import numpy as np              # For numerical operations
from sklearn.datasets import load_wine  # Built-in multiclass dataset
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.model_selection import train_test_split # To split dataset
from sklearn.metrics import classification_report    # Model evaluation report

# -------------------------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------------------------

wine = load_wine()
# Loads Wine dataset -> 178 samples, 13 chemical features, 3 wine classes

x = wine.data
# Feature matrix (Independent variables)
# Shape = (178 rows, 13 columns)

y = wine.target
# Target vector (Dependent variable)
# Values = 0, 1, 2 → representing 3 wine cultivars

print("Dataset Shape:", x.shape)
print("Target Classes:", wine.target_names)


# --------------------------------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,          # 30% data for testing
    random_state=42,        # Ensures same split every time
    stratify=y              # Maintains equal class distribution
)

print("Training Samples:", x_train.shape[0])
print("Testing Samples:", x_test.shape[0])

#-----------------------------------------------------------
#3.Create and Train Multinomial Logistic regression Model
#-----------------------------------------------------------

model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000
)
model.fit(x_train, y_train)
#Model learns weight coefficients for each of the 3 classes


#-----------------------------------------------------------
#4. Make Predictions
#-----------------------------------------------------------

y_pred = model.predict(x_test)
#predictsclass labels(0,1,2)

y_proba = model.predict_proba(x_test)
#predicts probability distribution for each class
#Example output:[0.01,0.95,0.04]

print(y_proba)
#--------------------------------------------------------------
#5.Evaluate Model Performance
#------------------------------------------------------------
print("Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=wine.target_names
))
# Shows precision, recall, f1-score for each class
'''
1 What is "Support"?

Support = number of actual samples in each class.

Class    Support
class_0  18 samples
class_1  21 samples
class_2  15 samples

So:
    
    Total = 18 + 21 + 15 = 54 samples

This means dataset is fairly balanced.

2 Understanding Precision  
Formula:  
Precision=True Positive / Predicted Positives  

Meaning:  
Out of all predicted as class_X, how many were actually c  
Example:  
For class_0 → Precision = 0.95  
That means:
    When model predicted class_0,  
95% of those predictions were correct.  
So only 5% were wrong predictions.  

3 Understanding Recall  
Recall=True Positives/Actual Positives))
    
Meaning:
Out of all actual class_X samples, how many did the model correctly classify?

Example:
For class_0 -> Recall = 1.00
That means:

There were 18 actual class_0 samples

Model correctly identified all 18

It did not miss any class_0 sample

Very strong performance.

4. Understanding F1-Score

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

It balances precision and recall.

For class_0:

Precision = 0.95

Recall = 1.00

F1 = 0.97

High F1 means model is both accurate and complete.

Now Let's Analyze Class-wise

class_0

Metric	Value	Meaning
class_0

Metric	Value	Meaning
Precision	0.95	Few false alarms
Recall	1.00	No missed samples
F1	0.97	Excellent
Support	18	Sample size
Interpretation:
Model perfectly captured all class_0 samples and made very few mistakes when predicting class_1

class_1

Metric	Value
Precision	0.95
Recall	0.95
F1	0.95
Support	21
Interpretation:

Slightly lower than class_0

Some small confusion exists

Balanced performance

class_2

Metric	Value
Precision	1.00
Recall	0.93
F1	0.97
Support	15
Important observation:

Precision = 1.00

Recall = 0.93

Meaning:

When model predicts class_2 → always correct

But it missed some actual class_2 samples

So:

Model is conservative for class_2

It does not falsely label other classes as class_2

But it misses a few true class_2 samples

Overall Accuracy

accuracy = 0.96

Meaning:
Out of 54 test samples:
0.96 × 54 ≈ 52

0.96 × 54 = 52 correct predictions

Only about 2 mistakes total

Very strong model performance.

This model:

Is not overfitting

Is not biased toward any class

Has balanced precision & recall

Performs excellent multiclass classification
'''

#-----------------------------------------------------------
#step6:Manual Class-wise Accuracy Calculation
#-----------------------------------------------------------

print("Class-wise Accuracy")

for i, class_name in enumerate(wine.target_names):
    
   class_indices = (y_test == i)
    # Select all test samples belonging to class i

   correct_predictions = np.sum(y_pred[class_indices] == y_test[class_indices])
    # Count correct predictions for that class

   total_samples = np.sum(class_indices)
    # Total samples of that class

   class_accuracy = correct_predictions / total_samples

   print(f"{class_name} Accuracy: {class_accuracy:.2f}")