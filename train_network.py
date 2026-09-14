"""Train a small ReLU network using the same prepared data as the linear model."""

from copy import deepcopy
from pathlib import Path

import torch

import prepare_data as data


def make_model(input_count, hidden_count):
    # Each hidden neuron sees all inputs and has its own weights and bias.
    return torch.nn.Sequential(
        torch.nn.Linear(input_count, hidden_count),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_count, 1),
    )


def rmse(model, inputs, ratings):
    with torch.no_grad():
        predictions = model(inputs).squeeze(-1)
        return (predictions - ratings).square().mean().sqrt().item()


def main():
    torch.manual_seed(42)
    hidden_count = 8
    learning_rate = 0.001
    epochs = 100

    model = make_model(data.X_train_tensor.shape[1], hidden_count)
    # Keep random weights so hidden neurons can learn different features.
    # Put the initial output near the training mean, rather than near zero.
    with torch.no_grad():
        model[2].bias.fill_(data.baseline_rating.item())

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_parameters = deepcopy(model.state_dict())
    model.eval()
    best_rmse = rmse(model, data.X_validation_tensor, data.y_validation_tensor)
    best_epoch = 0
    history = []
    print("\nNetwork:", model)
    print("Learned parameters:", sum(p.numel() for p in model.parameters()))
    print(f"Initial validation RMSE: {best_rmse:.4f}")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_inputs, batch_ratings in data.training_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_inputs).squeeze(-1)
            loss = (predictions - batch_ratings).square().mean()
            loss.backward()
            optimizer.step()

        model.eval()
        training_rmse = rmse(model, data.X_train_tensor, data.y_train_tensor)
        validation_rmse = rmse(model, data.X_validation_tensor, data.y_validation_tensor)
        if not (torch.isfinite(torch.tensor([training_rmse, validation_rmse])).all()):
            raise RuntimeError("Loss became nonfinite; inspect the learning rate and inputs.")
        history.append({"epoch": epoch, "training_rmse": training_rmse,
                        "validation_rmse": validation_rmse})
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_epoch = epoch
            best_parameters = deepcopy(model.state_dict())
        print(f"Epoch {epoch}: training RMSE = {training_rmse:.4f}, "
              f"validation RMSE = {validation_rmse:.4f}")

    model.load_state_dict(best_parameters)
    model.eval()
    restored_rmse = rmse(model, data.X_validation_tensor, data.y_validation_tensor)
    checkpoint = {
        "architecture": {"type": "relu_network", "input_count": model[0].in_features,
                         "hidden_count": hidden_count, "output_count": 1},
        "model_state": model.state_dict(),
        "input_columns": data.X_train.columns.tolist(),
        "numeric_columns": data.numeric_columns,
        "training_means": data.training_means.to_dict(),
        "training_stds": data.training_stds.to_dict(),
        "categories": data.categories,
        "best_epoch": best_epoch,
        "validation_rmse": restored_rmse,
        "history": history,
        "learning_rate": learning_rate,
        "seed": 42,
    }
    destination = Path(__file__).parent / "models" / "relu_rating_model.pt"
    destination.parent.mkdir(exist_ok=True)
    torch.save(checkpoint, destination)

    # Verify the on-disk checkpoint recreates the selected model's predictions.
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    reloaded = make_model(saved["architecture"]["input_count"], hidden_count)
    reloaded.load_state_dict(saved["model_state"])
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(data.X_validation_tensor),
                                   model(data.X_validation_tensor))
    print(f"\nRestored epoch {best_epoch}: validation RMSE = {restored_rmse:.4f}")
    print(f"Constant baseline validation RMSE: {data.baseline_rmse.item():.4f}")
    print("Saved and verified:", destination)


if __name__ == "__main__":
    main()
