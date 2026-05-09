import pandas as pd
import numpy as np
import copy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from fipe.discrepancy_oracle import DiscrepancyOracle
from fipe.model_prep import prepare_discrepancy_models
from fipe.feature import FeatureEncoder

print("Generating data...")
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training models...")

# Model A: Standard AdaBoost (Depth = 1)
model_a = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10,
    random_state=42
)
model_a.fit(X_train, y_train)

# Model B: AdaBoost with Depth = 2
model_b = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=10,
    random_state=42
)
model_b.fit(X_train, y_train)



print("Preparing Discrepancy Oracle inputs...")
combined_estimators, w_A, w_B = prepare_discrepancy_models(model_a, model_b)

df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
encoder = FeatureEncoder(df_train)


super_ensemble_model = copy.deepcopy(model_a)
super_ensemble_model.estimators_ = combined_estimators
if hasattr(super_ensemble_model, 'n_estimators'):
    super_ensemble_model.n_estimators = len(combined_estimators)

oracle = DiscrepancyOracle(
    base=super_ensemble_model,
    encoder=encoder,
    weights=w_A  # Initialize with Model A active
)

oracle.build()

print("Searching for discrepancy points...")
disagreements = []

training_features = df_train.columns.tolist()

for x_discrepancy in oracle.find_discrepancies(
    model2_weights=w_B, 
    model_A=model_a, 
    model_B=model_b,
    feature_names=training_features
):

    x_input = x_discrepancy[training_features].values.reshape(1, -1)
    
    pred_a = model_a.predict(x_input)[0]
    pred_b = model_b.predict(x_input)[0]
    
    if pred_a != pred_b:
        print("\n" + "="*40)
        print(f">> CONFIRMED DISCREPANCY")
        print("="*40)
        print(f"Input:\n{x_discrepancy}")
        print(f"Model A (Depth 1): {pred_a}")
        print(f"Model B (Depth 2): {pred_b}")
        disagreements.append(x_discrepancy)

print(f"\nTotal discrepancy points found: {len(disagreements)}")