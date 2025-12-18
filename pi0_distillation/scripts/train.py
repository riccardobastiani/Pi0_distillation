import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import TRMPolicyFiLM
from src.dataset import DistilledTeacherDataset

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 50
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(_ROOT_DIR, "data", "distilled")
SAVE_PATH = os.path.join(_ROOT_DIR, "outputs", "student_model.pt")


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DistilledTeacherDataset(DATA_DIR)
    if len(dataset) == 0:
        print("Dataset is empty. Skipping training.")
        return

    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    model = TRMPolicyFiLM().to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()

    print("Starting training loop ...")
    for epoch in range(EPOCHS):
        model.train()
        losses: list[float] = []
        for batch in train_dl:
            obs = batch["observations"].to(device)
            prop = batch["proprio_states"].to(device)
            act = batch["actions"].to(device)
            prompts = batch["prompt"]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(obs, prop, prompts)
                loss = F.smooth_l1_loss(pred, act)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), SAVE_PATH)

    model.eval()
    with torch.no_grad():
        val_losses: list[float] = []
        for batch in val_dl:
            obs = batch["observations"].to(device)
            prop = batch["proprio_states"].to(device)
            act = batch["actions"].to(device)
            prompts = batch["prompt"]
            pred = model(obs, prop, prompts)
            val_losses.append(F.smooth_l1_loss(pred, act).item())
        print(f"Validation loss: {sum(val_losses) / len(val_losses):.4f}")


if __name__ == "__main__":
    train()
