# src/training/train.py

import torch
from torch.utils.data import DataLoader

from src.data.scanpath_dataset import ScanpathSequenceDataset
from src.models.lstm_predictor import SAEPredictor
from src.training.loss import NLLLoss


def main():
    # ------------------------------------------------------------
    # 1. Dataset & DataLoader
    # ------------------------------------------------------------
    train_ds = ScanpathSequenceDataset(
        root="data/processed",
        npy_name="quaternions_sequence0.npy",
        seq_len=120,   # 120 timesteps per sample (≈2 seconds at 60 Hz)
    )

    # Infer feature_dim from one sample
    sample = train_ds[0]
    feature_dim = sample["features"].shape[-1]

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=0,   # keep 0 so it works cleanly on macOS / notebooks
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------------------------------------------
    # 2. Model, loss function, optimizer
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SAEPredictor(feature_dim=feature_dim).to(device)
    loss_fn = NLLLoss(mae_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # ------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["features"].to(device)         # (B, T, F)
            y_250 = batch["target_250"].to(device)   # (B, T, 3)
            y_500 = batch["target_500"].to(device)   # (B, T, 3)
            y_1000 = batch["target_1000"].to(device) # (B, T, 3)

            optimizer.zero_grad()

            out_250, out_500, out_1000 = model(x)    # each (B, T, 4)

            loss = (
                loss_fn(out_250, y_250)
                + loss_fn(out_500, y_500)
                + loss_fn(out_1000, y_1000)
            ) / 3.0

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    # ------------------------------------------------------------
    # 4. Save model
    # ------------------------------------------------------------
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    main()
