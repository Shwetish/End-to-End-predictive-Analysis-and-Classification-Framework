#=================================================================
#Business understanding
#================================================================
1. Business Problem Statement:-
The computer market changes quickly. Retailers and manufacturers
 do not know exactly how much value customers place on specific 
 features (like more RAM vs. a bigger screen vs. a faster
 processor), making it hard to price computers 
 competitively.

2. Business Objective
To understand how different hardware specifications affect the
 final cost and predict the market price of a computer based on
 its features.

3. Motivation
Smart Pricing Strategy: Computer manufacturers can set the 
perfect retail price for new laptops/desktops to maximize sales.

Cost vs. Feature Optimization: Companies can figure out which
 parts (like a CD drive or multi-core support) add the most 
 value, helping them design better, higher-profit computers.

4. Constraints
Rapid Tech Depreciation: Computer parts lose value very quickly
 over time (represented by the trend column), meaning pricing 
 models must constantly update.

Missing Details: The dataset lacks brand names (e.g., Apple vs.
Dell) and specific processor generations, which significantly 
influence customer willingness to pay.

5. Business Success Criteria
Computer brands can price new models accurately, leading to
 faster inventory sales and fewer unsold items.

Production teams can cut costs by removing features that 
customers dont care to pay extra for.

6. ML Success Criteria
Build a machine learning model (like regression) that accurately 
estimates computer price based on its hardware specifications.

Achieve high prediction accuracy, aiming for an R 
2 score of 0.85 or higher to reliably capture price trends.

