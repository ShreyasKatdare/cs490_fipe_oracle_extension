import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def verify_discrepancies():
    # ---------------------------------------------------------
    # 1. RECREATE THE EXACT MODELS
    # ---------------------------------------------------------
    print("Recreating models with fixed seed...")
    # MUST use the same seed as run_experiment.py
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model A: Depth 1 (The "Weak" Learner)
    model_a = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=10,
        random_state=42
    )
    model_a.fit(X_train, y_train)

    # Model B: Depth 2 (The "Strong" Learner)
    model_b = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=10,
        random_state=42
    )
    model_b.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 2. TEST SPECIFIC POINTS FROM YOUR LOGS
    # ---------------------------------------------------------
    # These points were copied from your last successful run (Run 4).
    # Point 1: (Model A says 0, Model B says 1)
    p1 = [
        3.702647,  # feature_0
       -2.254602,  # feature_1
        1.222665,  # feature_2
       -2.718759,  # feature_3
        0.000000,  # feature_4
       -2.235716,  # feature_5
        0.000000,  # feature_6
       -1.299832,  # feature_7
        2.804758,  # feature_8
       -1.832252   # feature_9
    ]

    # Point 2: (Model A says 1, Model B says 0)
    p2 = [
       -2.899651, # feature_0
       -2.254602, # feature_1
        1.222665, # feature_2
        2.050649, # feature_3
        0.000000, # feature_4
        2.764060, # feature_5
        0.000000, # feature_6
        0.735096, # feature_7
        2.804758, # feature_8
        0.167748  # feature_9
    ]

    points_to_test = [("Point 1", p1), ("Point 2", p2)]
    feature_names = [f"feature_{i}" for i in range(10)]

    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)

    for name, point in points_to_test:
        # Reshape for sklearn (1 sample, 10 features)
        x_input = np.array(point).reshape(1, -1)
        
        # Get predictions
        pred_a = model_a.predict(x_input)[0]
        pred_b = model_b.predict(x_input)[0]
        
        # Get probabilities (confidence)
        prob_a = model_a.predict_proba(x_input)[0]
        prob_b = model_b.predict_proba(x_input)[0]

        print(f"\nTesting {name}:")
        print(f"Input: {point}")
        print("-" * 30)
        print(f"Model A (Depth 1): Class {pred_a}  (Conf: {prob_a})")
        print(f"Model B (Depth 2): Class {pred_b}  (Conf: {prob_b})")
        
        if pred_a != pred_b:
            print(">> DISCREPANCY CONFIRMED ✅")
        else:
            print(">> MODELS AGREE ❌ (Something changed)")

if __name__ == "__main__":
    verify_discrepancies()