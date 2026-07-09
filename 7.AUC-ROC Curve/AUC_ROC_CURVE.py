import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Data from the image table
classes = ['P','P','N','P','P','N','N','N','P','N']
probs   = [0.90,0.80,0.70,0.60,0.55,0.54,0.53,0.51,0.50,0.40]

# Convert classes to binary labels: P=1, N=0
y_true = np.array([1 if c=='P' else 0 for c in classes])
y_scores = np.array(probs)

# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

# Find sweet spot using Youden's J statistic
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]

print(f"AUC: {auc:.2f}")
print(f"Sweet Spot Threshold: {best_threshold:.2f}")
print(f"TPR at sweet spot: {best_tpr:.2f}")
print(f"FPR at sweet spot: {best_fpr:.2f}")

# Plot ROC curve
plt.figure(figsize=(7,7))
plt.plot(fpr, tpr, marker='o', label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0,1],[0,1],'k--', label="Random Guess")

# Mark sweet spot
plt.scatter(best_fpr, best_tpr, s=120, color='red', zorder=5, label="Sweet Spot")
plt.text(best_fpr+0.02, best_tpr-0.05, f"T={best_threshold:.2f}", color='red', fontsize=10)

plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve with Sweet Spot (Youden's J)")
plt.legend()
plt.grid(True)
plt.show()
#################################
#Justification how the sweet spot is selected
'''
What Is the “Sweet Spot”?
In ROC analysis, the sweet spot is:
The threshold that maximizes separation between positive and negative classes.
Graphically:
The point farthest from the diagonal (random guess line)
The point closest to top-left corner (0,1)

Youden’s J Statistic:
    J=TPR−FPR
    
    if TPR is high AND FPR is low → Good model

    TPR and FPR both high → Risky model

    So maximizing:TPR−FPR

means:Increase true detection Penalize false detection That balance point is your 
sweet spot.

Geometric Interpretation

The diagonal line represents:

TPR=FPR

TPR−FPR=0
At sweet spot:
TPR−FPR=maximum
Meaning:
The vertical distance between ROC curve and diagonal is maximum.

That is exactly what Youden’s J finds.

'''
import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------------------
# Step 1: Input data
# ---------------------------
# 'classes' are the actual outcomes: P = Positive, N = Negative
# 'probs' are the predicted probabilities from the model
classes = ['P','P','N','P','P','N','N','N','P','N']
probs   = [0.90,0.80,0.70,0.60,0.55,0.54,0.53,0.51,0.50,0.40]

# Convert P/N to binary form: 1 for Positive, 0 for Negative
y_true = np.array([1 if c=='P' else 0 for c in classes])
y_scores = np.array(probs)

# ---------------------------
# Step 2: Define thresholds to test
# ---------------------------
# These thresholds represent the decision boundary:
# If predicted probability >= threshold → classify as Positive
# Else → classify as Negative
thresholds_to_check = [0.90, 0.80, 0.70, 0.60, 0.55, 0.54, 0.53, 0.51, 0.50, 0.40]

# Print table headers
print(f"{'Thr':>6} {'TPR':>6} {'FPR':>6} {'J=TPR-FPR':>10}")
print("-"*32)
'''
'Thr':>6 → Print the word "Thr" (for Threshold) right-aligned in a field of width 6 
 characters.

'TPR':>6 → Same idea for True Positive Rate.

'FPR':>6 → Same for False Positive Rate.

'J=TPR-FPR':>10 → Print the column name “J=TPR-FPR” right-aligned in 10 spaces.

'''
best_thr, best_j = None, -1  # To track best threshold and best Youden's J

# ---------------------------
# Step 3: Loop over thresholds
# ---------------------------
for thr in thresholds_to_check:
    # Classify based on threshold
    y_pred = (y_scores >= thr).astype(int)
    
    # Confusion matrix gives: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # True Positive Rate (Sensitivity, Recall)
    tpr = tp / (tp + fn)
    
    # False Positive Rate = 1 - Specificity
    fpr = fp / (fp + tn)
    
    # Youden's J statistic = TPR - FPR
    # Higher J means better trade-off between Sensitivity and Specificity
    j = tpr - fpr
    
    # Show results for this threshold
    print(f"{thr:>6.2f} {tpr:>6.2f} {fpr:>6.2f} {j:>10.2f}")
    
    # Keep track of the best threshold based on maximum J
    if j > best_j:
        best_j = j
        best_thr = thr

# ---------------------------
# Step 4: Display best threshold
# ---------------------------
print("\nBest Threshold:", best_thr, "with Youden's J =", round(best_j, 2))

# ---------------------------
# Educational Notes:
# ---------------------------
# 1. At very high thresholds (e.g., 0.90), TPR is low → many positives missed.
# 2. At very low thresholds (e.g., 0.40), TPR is high but FPR also becomes high → many false alarms.
# 3. The "sweet spot" is where TPR is high AND FPR is low.
# 4. Youden's J helps us find this sweet spot numerically.


