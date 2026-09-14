"""Final evaluation of the frozen model on the reserved test set.

This is the one place the test set is allowed to be touched, and only for
reporting a final error metric. Run once after training and export are done;
never used to pick epochs, tune hyperparameters, or select factors.
"""
from pathlib import Path

import torch

import prepare_data as data
from predict import load_model


def report(name, predictions, targets):
    errors = predictions - targets
    mae = errors.abs().mean().item()
    rmse = errors.square().mean().sqrt().item()
    ss_res = errors.square().sum().item()
    ss_tot = (targets - targets.mean()).square().sum().item()
    r2 = 1 - ss_res / ss_tot
    print(f"{name}: MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    artifact_path = Path(__file__).parent / "artifacts" / "relu_rating_model.json"
    model, checkpoint = load_model(artifact_path)

    with torch.no_grad():
        baseline_predictions = torch.full_like(data.y_test_tensor, data.baseline_rating.item())
        model_predictions = model(data.X_test_tensor).squeeze(-1)

    print(f"Test set: {len(data.y_test_tensor)} ratings from {data.test['ID'].nunique()} skiers")
    print(f"Mean training rating (the baseline's constant prediction): {data.baseline_rating.item():.2f}")
    print()
    report("Mean-rating baseline", baseline_predictions, data.y_test_tensor)
    report("Saved ReLU network", model_predictions, data.y_test_tensor)


if __name__ == "__main__":
    main()
