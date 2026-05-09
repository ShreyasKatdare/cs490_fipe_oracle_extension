from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split

from fipe import FIPE, FeatureEncoder
from fipe.env import ENV

ENV.pruner_solver = "gurobi"
DATASET_NAME = "FICO"

MODEL_TYPE = "AdaBoost"

N_ESTIMATORS = 50

SEED = 42

NORM = 1
# ===========================================


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a dataset from the tests/datasets-for-tests folder."""
    dataset_path = Path(__file__).parent / "tests" / "datasets-for-tests" / dataset_name
    data = pd.read_csv(dataset_path / f"{dataset_name}.full.csv")
    
    # Last column is the label
    labels = data.iloc[:, -1]
    y = labels.astype("category").cat.codes.to_numpy().ravel()
    X = data.iloc[:, :-1]
    
    return X, y


def train_model(model_type: str, X_train, y_train, n_estimators: int, seed: int):
    """Train an ensemble model."""
    if model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            random_state=seed,
        )
    elif model_type == "AdaBoost":
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=seed,
        )
    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            init="zero",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model


def get_weights(model, n_estimators: int) -> np.ndarray:
    """Get weights for the ensemble trees."""
    if isinstance(model, AdaBoostClassifier):
        weights = model.estimator_weights_.copy()
    else:
        weights = np.ones(n_estimators)
    
    weights = (weights / weights.max()) * 1e5
    return weights


def main():
    print("=" * 60)
    print(f"FIPE: Functionally Identical Pruning of Ensembles")
    print("=" * 60)
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Model: {MODEL_TYPE} with {N_ESTIMATORS} estimators")
    print(f"Norm: L{NORM}")
    print("-" * 60)

    print("\n[1/5] Loading dataset...")
    X, y = load_dataset(DATASET_NAME)
    print(f"      Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"      Classes: {np.unique(y)}")

    print("\n[2/5] Encoding features...")
    encoder = FeatureEncoder(X)
    X_encoded = encoder.X.to_numpy()
    print(f"      Encoded shape: {X_encoded.shape}")
    print(f"      Binary features: {len(encoder.binary)}")
    print(f"      Categorical features: {len(encoder.categorical)}")
    print(f"      Continuous features: {len(encoder.continuous)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=SEED
    )

    print(f"\n[3/5] Training {MODEL_TYPE} with {N_ESTIMATORS} trees...")
    model = train_model(MODEL_TYPE, X_train, y_train, N_ESTIMATORS, SEED)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_acc = np.mean(y_pred_train == y_train) * 100
    test_acc = np.mean(y_pred_test == y_test) * 100
    print(f"      Original model - Train accuracy: {train_acc:.2f}%")
    print(f"      Original model - Test accuracy: {test_acc:.2f}%")

    weights = get_weights(model, N_ESTIMATORS)

    env = gp.Env()
    env.setParam("OutputFlag", 0)

    print(f"\n[4/5] Building FIPE pruner (L{NORM} norm)...")
    pruner = FIPE(
        base=model,
        encoder=encoder,
        weights=weights,
        norm=NORM,
        env=env,
        eps=1e-6,
        tol=1e-4,
    )
    pruner.build()
    pruner.add_samples(X_train)

    print("\n[5/5] Pruning ensemble...")
    print("      (This may take a while for large ensembles)")
    pruner.prune()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    n_active = pruner.n_active_estimators
    reduction = (1 - n_active / N_ESTIMATORS) * 100
    
    print(f"\nPruning Results:")
    print(f"  - Original trees: {N_ESTIMATORS}")
    print(f"  - Active trees:   {n_active}")
    print(f"  - Reduction:      {reduction:.1f}%")
    
    y_pred_original = model.predict(X_test)
    y_pred_pruned = pruner.predict(X_test)
    fidelity = np.mean(y_pred_original == y_pred_pruned) * 100
    
    print(f"\nFidelity Check (on test set):")
    print(f"  - Original model predictions == Pruned model predictions")
    print(f"  - Fidelity: {fidelity:.2f}%")
    
    pruned_acc = np.mean(y_pred_pruned == y_test) * 100
    print(f"\nAccuracy Comparison:")
    print(f"  - Original model: {test_acc:.2f}%")
    print(f"  - Pruned model:   {pruned_acc:.2f}%")
    print(f"\nOracle Statistics:")
    print(f"  - Number of oracle calls: {pruner.n_oracle_calls}")
    print(f"  - Counter-examples found: {sum(len(x) for x in pruner.oracle_samples)}")
    
    print("\n" + "=" * 60)
    print("The pruned model is GUARANTEED to be functionally identical")
    print("to the original model on the ENTIRE feature space!")
    print("=" * 60)


if __name__ == "__main__":
    main()
