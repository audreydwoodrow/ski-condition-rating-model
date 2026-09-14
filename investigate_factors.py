"""Retrain with one factor hidden; use validation only, never test outcomes."""
import json
import math
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_network import data, make_model, rmse


def main():
    torch.set_num_threads(1)
    columns = data.X_train.columns.tolist()
    factors = {
        "Queue time": ["QUE_TIME"],
        "Price": ["PRICE"],
        "Temperature": ["TEMPERATURE"],
        "Slopes open": ["SLOPES_OPEN"],
        **{name: [c for c in columns if c.startswith(prefix + "_")]
           for name, prefix in [("Weather", "WEATHER"), ("Wind", "WIND"),
                                ("Day type", "WEKKDAY"), ("Vacation period", "PERIOD")]},
    }
    seeds = [42, 43, 44]
    results = []
    for seed in seeds:
        for factor, hidden_columns in {"All factors": [], **factors}.items():
            train_x = data.X_train_tensor.clone()
            validation_x = data.X_validation_tensor.clone()
            indexes = [columns.index(c) for c in hidden_columns]
            # Constant zero removes variation/information, preserving the same
            # architecture and identical initial parameters across comparisons.
            train_x[:, indexes] = 0
            validation_x[:, indexes] = 0
            torch.manual_seed(seed)
            model = make_model(len(columns), 8)
            with torch.no_grad():
                model[2].bias.fill_(data.baseline_rating.item())
            loader = DataLoader(TensorDataset(train_x, data.y_train_tensor),
                                batch_size=32, shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            best = rmse(model, validation_x, data.y_validation_tensor)
            best_epoch = 0
            for epoch in range(1, 101):
                model.train()
                for x, y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(x).squeeze(-1) - y).square().mean()
                    loss.backward()
                    optimizer.step()
                model.eval()
                score = rmse(model, validation_x, data.y_validation_tensor)
                if not math.isfinite(score):
                    raise RuntimeError(f"Nonfinite loss for {factor}, seed {seed}")
                if score < best:
                    best, best_epoch = score, epoch
            results.append(dict(seed=seed, factor=factor, rmse=best,
                                best_epoch=best_epoch, hidden_columns=hidden_columns))
            print(f"seed={seed}, hidden={factor}, best RMSE={best:.4f}, epoch={best_epoch}", flush=True)

    baselines = {r["seed"]: r["rmse"] for r in results if r["factor"] == "All factors"}
    summary = []
    for factor in factors:
        rows = [r for r in results if r["factor"] == factor]
        changes = [r["rmse"] - baselines[r["seed"]] for r in rows]
        summary.append(dict(factor=factor, mean_rmse=mean(r["rmse"] for r in rows),
                            mean_change=mean(changes), min_change=min(changes),
                            max_change=max(changes)))
    summary.sort(key=lambda r: r["mean_change"], reverse=True)
    report = dict(seeds=seeds, epochs=100, learning_rate=0.001, batch_size=32,
                  all_factors_mean=mean(baselines.values()), results=results, summary=summary)
    destination = Path(__file__).parent / "models" / "factor_investigation.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_factors_mean": report["all_factors_mean"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
