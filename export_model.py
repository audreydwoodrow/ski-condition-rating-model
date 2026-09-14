"""Export the locally trained network as a portable JSON prediction artifact."""
import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).parent
    checkpoint = torch.load(root / "models" / "relu_rating_model.pt",
                            map_location="cpu", weights_only=True)
    checkpoint.pop("history", None)
    checkpoint["model_state"] = {
        key: value.tolist() for key, value in checkpoint["model_state"].items()
    }
    destination = root / "artifacts" / "relu_rating_model.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(checkpoint, indent=2) + "\n")
    print("Exported:", destination)


if __name__ == "__main__":
    main()
