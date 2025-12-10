# src/training/train.py

import torch
from torch.utils.data import DataLoader

from src.data.processed_npy_dataset import ProcessedNPYDataset
from src.data.collate_fn import pad_collate
from src.models.lstm_predictor import SAEPredictor
from src.training.loss import NLLLoss


def main():

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    data_root = "data/processed"   # folder containing sequence_*.npy files

    train_ds = ProcessedNPYDataset(data_root, split="train")

    # infer feature_dim from a sample
    sample = train_ds[0]
    feature_dim = sample["features"].shape[-1]

    # DataLoader WITH padding
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate,   # ← IMPORTANT
    )

    # --------------------------------------------------------
    # 2. Model + Loss + Optimizer
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SAEPredictor(feature_dim=feature_dim).to(device)
    loss_fn = NLLLoss(mae_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # --------------------------------------------------------
    # 3. Training Loop
    # --------------------------------------------------------
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch["features"].to(device)       # (B, T, F)
            y250 = batch["target_250"].to(device)  # (B, T, 3)
            y500 = batch["target_500"].to(device)
            y1000 = batch["target_1000"].to(device)

            optimizer.zero_grad()

            out250, out500, out1000 = model(x)     # (B, T, 4)

            # average across three horizons
            loss = (
                loss_fn(out250, y250) +
                loss_fn(out500, y500) +
                loss_fn(out1000, y1000)
            ) / 3.0

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg:.4f}")

    # --------------------------------------------------------
    # 4. Save model
    # --------------------------------------------------------
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    main()
