from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from pathlib import Path
import pandas as pd

data_path = Path(__file__).parent / "data" / "DATA.xlsx"
df = pd.read_excel(data_path)

# The survey (Haugom, Malasevska & Lien) was fielded at Hafjell in February 2018.
# Convert to Fahrenheit and USD once, right after loading, so every downstream
# script (training, diagnostics, the exported artifact) works in those units.
# Rate: Norges Bank official monthly average USD/NOK for February 2018.
NOK_PER_USD = 7.8327

df["TEMPERATURE"] = df["TEMPERATURE"] * 9 / 5 + 32
df["PRICE"] = df["PRICE"] / NOK_PER_USD

# Convert valid ratings to numbers; mark nonnumeric answers as missing.
df["RATING"] = pd.to_numeric(df["RATING"], errors="coerce")

# Keep genuine zeros, but exclude unanswered or invalid ratings.
clean = df.dropna(subset=["RATING"]).copy()

print("Original rows:", len(df))
print("Rows with numeric ratings:", len(clean))
print("Distinct skiers remaining:", clean["ID"].nunique())

# Get each skier's ID once.
skier_ids = clean["ID"].unique()

# Shuffle reproducibly.
rng = np.random.default_rng(seed=42)
rng.shuffle(skier_ids)

# Find the boundaries between groups.
train_end = int(0.70 * len(skier_ids))
validation_end = train_end + int(0.15 * len(skier_ids))

train_ids = skier_ids[:train_end]
validation_ids = skier_ids[train_end:validation_end]
test_ids = skier_ids[validation_end:]

# Select all responses belonging to each group.
train = clean[clean["ID"].isin(train_ids)].copy()
validation = clean[clean["ID"].isin(validation_ids)].copy()
test = clean[clean["ID"].isin(test_ids)].copy()

for name, group in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test),
]:
    print(
        f"{name}: {group['ID'].nunique()} skiers, "
        f"{len(group)} examples"
    )

assert set(train_ids).isdisjoint(validation_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(validation_ids).isdisjoint(test_ids)

assert len(train) + len(validation) + len(test) == len(clean)

numeric_columns = [
    "QUE_TIME",
    "PRICE",
    "TEMPERATURE",
    "SLOPES_OPEN",
]

# Learn the scaling values from training inputs only.
training_means = train[numeric_columns].mean()
training_stds = train[numeric_columns].std(ddof=0)

assert (training_stds > 0).all()

# Apply the same transformation to all three groups.
train_numeric = (
    train[numeric_columns] - training_means
) / training_stds

validation_numeric = (
    validation[numeric_columns] - training_means
) / training_stds

test_numeric = (
    test[numeric_columns] - training_means
) / training_stds

print("\nScaling values:")
print(pd.DataFrame({
    "training_mean": training_means,
    "training_std": training_stds,
}))

print("\nFirst training example, before scaling:")
print(train[numeric_columns].iloc[0])

print("\nSame example, after scaling:")
print(train_numeric.iloc[0])

categories = {
    "WEATHER": ["SUN", "CLOUDY", "FOG", "SNOW", "RAIN"],
    "WIND": ["NO WIND", "GENTLE BREEZE", "FRESH BREEZE"],
    "WEKKDAY": ["MIDWEEK", "WEEKEND"],
    "PERIOD": ["REGULAR WEEK", "VACATION"],
}


def encode_categories(rows):
    encoded = pd.DataFrame(index=rows.index)

    for column, allowed_values in categories.items():
        # Catch missing or unexpected categories.
        assert rows[column].isin(allowed_values).all(), column

        for value in allowed_values:
            encoded[f"{column}_{value}"] = (
                rows[column] == value
            ).astype(float)

    return encoded


train_categories = encode_categories(train)
validation_categories = encode_categories(validation)
test_categories = encode_categories(test)

# Join numeric and categorical inputs side by side.
X_train = pd.concat([train_numeric, train_categories], axis=1)
X_validation = pd.concat(
    [validation_numeric, validation_categories], axis=1
)
X_test = pd.concat([test_numeric, test_categories], axis=1)

print("\nTraining input shape:", X_train.shape)
print("\nFirst complete training input:")
print(X_train.iloc[0])

X_train_tensor = torch.tensor(
    X_train.to_numpy(), dtype=torch.float32
)
y_train_tensor = torch.tensor(
    train["RATING"].to_numpy(), dtype=torch.float32
)

X_validation_tensor = torch.tensor(
    X_validation.to_numpy(), dtype=torch.float32
)
y_validation_tensor = torch.tensor(
    validation["RATING"].to_numpy(), dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test.to_numpy(), dtype=torch.float32
)
y_test_tensor = torch.tensor(
    test["RATING"].to_numpy(), dtype=torch.float32
)

print("\nTraining inputs:", X_train_tensor.shape)
print("Training ratings:", y_train_tensor.shape)
print("First rating:", y_train_tensor[0].item())

# Pair each input row with its corresponding rating.
training_dataset = TensorDataset(
    X_train_tensor, y_train_tensor
)

# Use a repeatable sequence of random shuffles.
shuffle_generator = torch.Generator().manual_seed(42)

training_loader = DataLoader(
    training_dataset,
    batch_size=32,
    shuffle=True,
    generator=shuffle_generator,
)

# Inspect one full pass through the training examples.
batch_sizes = []

for batch_inputs, batch_ratings in training_loader:
    batch_sizes.append(len(batch_ratings))

    if len(batch_sizes) == 1:
        print("\nFirst batch inputs:", batch_inputs.shape)
        print("First batch ratings:", batch_ratings.shape)

print("Number of batches:", len(batch_sizes))
print("Last batch size:", batch_sizes[-1])
print("Total examples visited:", sum(batch_sizes))

# One constant prediction, calculated from training ratings only.
baseline_rating = y_train_tensor.mean()

# Evaluate that constant on validation examples.
with torch.no_grad():
    baseline_predictions = torch.full_like(
        y_validation_tensor,
        baseline_rating.item(),
    )

    baseline_mse = (
        (baseline_predictions - y_validation_tensor) ** 2
    ).mean()

    baseline_rmse = baseline_mse.sqrt()

print("\nBaseline rating:", baseline_rating.item())
print("Validation baseline MSE:", baseline_mse.item())
print("Validation baseline RMSE:", baseline_rmse.item())

# Importing this module prepares data; training runs only when executed directly.
if __name__ == "__main__":
    model = torch.nn.Linear(in_features=16, out_features=1)

    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(baseline_rating.item())

    with torch.no_grad():
        first_predictions = model(X_train_tensor[:3]).squeeze(-1)

    print("\nInitial predictions:", first_predictions)

    learning_rate = 0.001
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate
    )

    model.train()

    best_validation_rmse = float("inf")
    best_epoch = None
    best_parameters = None

    for epoch in range(100):
        model.train()

        for batch_inputs, batch_ratings in training_loader:
            optimizer.zero_grad(set_to_none=True)

            predictions = model(batch_inputs).squeeze(-1)
            loss = ((predictions - batch_ratings) ** 2).mean()

            loss.backward()
            optimizer.step()

        # Evaluate after every training batch has been processed.
        model.eval()

        with torch.no_grad():
            training_predictions = model(
                X_train_tensor
            ).squeeze(-1)

            training_rmse = (
                (training_predictions - y_train_tensor)
                .square()
                .mean()
                .sqrt()
            )

            validation_predictions = model(
                X_validation_tensor
            ).squeeze(-1)

            validation_rmse = (
                (validation_predictions - y_validation_tensor)
                .square()
                .mean()
                .sqrt()
            )

        current_validation_rmse = validation_rmse.item()

        if current_validation_rmse < best_validation_rmse:
            best_validation_rmse = current_validation_rmse
            best_epoch = epoch + 1
            best_parameters = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1}: "
            f"training RMSE = {training_rmse.item():.4f}, "
            f"validation RMSE = {validation_rmse.item():.4f}"
        )

    model.load_state_dict(best_parameters)

    print(
        f"\nRestored epoch {best_epoch}, "
        f"validation RMSE = {best_validation_rmse:.4f}"
    )

    print("\nBaseline validation RMSE:", baseline_rmse.item())

    model.eval()

    with torch.no_grad():
        restored_predictions = model(
            X_validation_tensor
        ).squeeze(-1)

        restored_rmse = (
            (restored_predictions - y_validation_tensor)
            .square()
            .mean()
            .sqrt()
        )

    print("\nSaved best RMSE:", best_validation_rmse)
    print("Recalculated restored RMSE:", restored_rmse.item())

    checkpoint = {
        "model_state": model.state_dict(),
        "input_columns": X_train.columns.tolist(),
        "numeric_columns": numeric_columns,
        "training_means": training_means.to_dict(),
        "training_stds": training_stds.to_dict(),
        "categories": categories,
        "best_epoch": best_epoch,
        "validation_rmse": restored_rmse.item(),
    }

    model_directory = Path(__file__).parent / "models"
    model_directory.mkdir(exist_ok=True)

    checkpoint_path = model_directory / "linear_rating_model_lr001.pt"
    torch.save(checkpoint, checkpoint_path)

    print("\nSaved model to:", checkpoint_path)
