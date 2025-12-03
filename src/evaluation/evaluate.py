# src/evaluation/evaluate.py

import torch
from torch.utils.data import DataLoader

from src.models.lstm_predictor import SAEPredictor
from src.data.processed_npy_dataset import ProcessedNPYDataset
from src.evaluation.metrics import great_circle_mae, nll_vmf


def evaluate_split(model, loader, device):
    mae_250, mae_500, mae_1000 = [], [], []
    nll_250, nll_500, nll_1000 = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)

            # Targets
            y250 = batch["target_250"].to(device)
            y500 = batch["target_500"].to(device)
            y1000 = batch["target_1000"].to(device)

            # Predictions
            p250, p500, p1000 = model(x)

            # Split (mu, kappa)
            mu250, k250 = p250[..., :3], p250[..., 3]
            mu500, k500 = p500[..., :3], p500[..., 3]
            mu1000, k1000 = p1000[..., :3], p1000[..., 3]

            # Great-Circle MAE (deg)
            mae_250.append(great_circle_mae(mu250, y250))
            mae_500.append(great_circle_mae(mu500, y500))
            mae_1000.append(great_circle_mae(mu1000, y1000))

            # NLL (nats)
            nll_250.append(nll_vmf(mu250, k250, y250))
            nll_500.append(nll_vmf(mu500, k500, y500))
            nll_1000.append(nll_vmf(mu1000, k1000, y1000))

    return {
        "MAE_250": sum(mae_250) / len(mae_250),
        "MAE_500": sum(mae_500) / len(mae_500),
        "MAE_1000": sum(mae_1000) / len(mae_1000),
        "NLL_250": sum(nll_250) / len(nll_250),
        "NLL_500": sum(nll_500) / len(nll_500),
        "NLL_1000": sum(nll_1000) / len(nll_1000),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SAEPredictor(feature_dim=6)  # xyz + omega
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)

    # Load Test Set
    test_ds = ProcessedNPYDataset(split="test")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Evaluate
    results = evaluate_split(model, test_loader, device)

    print("\n===== Evaluation Results (Test Set) =====")
    print(f"250ms:  MAE = {results['MAE_250']:.3f}°,   NLL = {results['NLL_250']:.3f}")
    print(f"500ms:  MAE = {results['MAE_500']:.3f}°,   NLL = {results['NLL_500']:.3f}")
    print(f"1000ms: MAE = {results['MAE_1000']:.3f}°,  NLL = {results['NLL_1000']:.3f}")


if __name__ == "__main__":
    main()
