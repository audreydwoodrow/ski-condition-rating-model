import torch

x = torch.tensor([
    [2.0, 1.0],  # Example A: queue, sunny
    [4.0, 0.0],  # Example B: queue, sunny
])

y = torch.tensor([60.0, 30.0])

learning_rate = 0.01

model = torch.nn.Linear(in_features=2, out_features=1)

with torch.no_grad():
    model.weight.copy_(torch.tensor([[-2.0, 10.0]]))
    model.bias.copy_(torch.tensor([50.0]))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for step in range(10):
    # Clear gradients from the previous step.
    optimizer.zero_grad(set_to_none=True)

    # Forward pass and loss.
    predictions = model(x).squeeze(-1)
    loss = ((predictions - y) ** 2).mean()

    # Backward pass.
    loss.backward()

    # Update the shared parameters.
    optimizer.step()

    # Check the result of the update.
    with torch.no_grad():
        updated_predictions = model(x).squeeze(-1)
        updated_loss = ((updated_predictions - y) ** 2).mean()

    print(
        f"Step {step + 1}: "
        f"loss {loss.item():.4f} -> {updated_loss.item():.4f}, "
        f"w={model.weight.detach().tolist()}, "
        f"b={model.bias.item():.4f}"
    )
