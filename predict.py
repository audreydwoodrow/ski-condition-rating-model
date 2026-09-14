"""Predict a willingness rating without loading survey data or training."""
import argparse
import json
import math
from pathlib import Path

import torch


def load_model(path):
    checkpoint = json.loads(path.read_text())
    architecture = checkpoint["architecture"]
    model = torch.nn.Sequential(
        torch.nn.Linear(architecture["input_count"], architecture["hidden_count"]),
        torch.nn.ReLU(),
        torch.nn.Linear(architecture["hidden_count"], 1),
    )
    model.load_state_dict({
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in checkpoint["model_state"].items()
    })
    model.eval()
    return model, checkpoint


def encode_scenario(scenario, checkpoint):
    encoded = {}
    for column in checkpoint["numeric_columns"]:
        value = float(scenario[column])
        if not math.isfinite(value):
            raise ValueError(f"{column} must be finite")
        encoded[column] = (value - checkpoint["training_means"][column]) / checkpoint["training_stds"][column]
    if scenario["QUE_TIME"] < 0 or scenario["PRICE"] < 0:
        raise ValueError("Queue time and price must be nonnegative")
    if not 0 <= scenario["SLOPES_OPEN"] <= 1:
        raise ValueError("Slopes open must be a fraction between 0 and 1")
    for column, categories in checkpoint["categories"].items():
        if scenario[column] not in categories:
            raise ValueError(f"Unknown {column}: {scenario[column]}")
        for category in categories:
            encoded[f"{column}_{category}"] = float(scenario[column] == category)
    values = [encoded[column] for column in checkpoint["input_columns"]]
    return torch.tensor([values], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=float, default=10, help="Queue minutes")
    parser.add_argument("--price", type=float, default=250, help="Day-pass price in Norwegian kroner")
    parser.add_argument("--temperature", type=float, default=-5, help="Degrees Celsius")
    parser.add_argument("--slopes-open", type=float, default=1, help="Fraction, e.g. 0.75")
    parser.add_argument("--weather", choices=["SUN", "CLOUDY", "FOG", "SNOW", "RAIN"], default="SUN")
    parser.add_argument("--wind", choices=["NO WIND", "GENTLE BREEZE", "FRESH BREEZE"], default="NO WIND")
    parser.add_argument("--day", choices=["MIDWEEK", "WEEKEND"], default="MIDWEEK")
    parser.add_argument("--period", choices=["REGULAR WEEK", "VACATION"], default="REGULAR WEEK")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "artifacts" / "relu_rating_model.json")
    args = parser.parse_args()
    scenario = dict(QUE_TIME=args.queue, PRICE=args.price, TEMPERATURE=args.temperature,
                    SLOPES_OPEN=args.slopes_open, WEATHER=args.weather, WIND=args.wind,
                    WEKKDAY=args.day, PERIOD=args.period)
    try:
        model, checkpoint = load_model(args.model)
        inputs = encode_scenario(scenario, checkpoint)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    with torch.no_grad():
        rating = model(inputs).item()
    print("Scenario:")
    for key, value in scenario.items():
        print(f"  {key}: {value}")
    print(f"\nPredicted willingness-to-ski rating: {rating:.2f}")
    print("Survey scale: 0–100. This is not a calibrated probability.")
    if not 0 <= rating <= 100:
        print("Raw prediction is outside the survey scale; the output is unbounded.")
    print(f"Saved validation RMSE: {checkpoint['validation_rmse']:.2f} rating points")
    print("No training or test-set evaluation was performed.")


if __name__ == "__main__":
    main()
