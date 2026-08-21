import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

print("MNIST indiriliyor (60.000 eğitim, 10.000 test görüntüsü)...")
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Eğitim örnekleri: {len(train_dataset)}, Test örnekleri: {len(test_dataset)}\n")


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x)


def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total * 100


def train_with_schedule(schedule_name, epochs=5, base_lr=0.01):
    torch.manual_seed(42)
    model = MNISTNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_steps = epochs * len(train_loader)

    if schedule_name == "constant":
        scheduler = None
    elif schedule_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    elif schedule_name == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=base_lr, total_steps=total_steps)
    elif schedule_name == "step":
        scheduler = StepLR(optimizer, step_size=total_steps // 4, gamma=0.3)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader)
        print(f"    Epoch {epoch+1}: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")

    elapsed = time.time() - start
    final_loss, final_acc = evaluate(model, test_loader)
    return final_loss, final_acc, elapsed


print("=== MNIST üzerinde gerçek ölçekli Schedule karşılaştırması (5 epoch, AdamW) ===\n")

results = {}
for name in ["constant", "cosine", "onecycle", "step"]:
    print(f"--- {name} ---")
    loss, acc, elapsed = train_with_schedule(name, epochs=5, base_lr=0.001)
    results[name] = (loss, acc, elapsed)
    print(f"  Final: loss={loss:.4f}, accuracy={acc:.2f}%, süre={elapsed:.1f}s\n")

print(f"\n{'Schedule':<12} {'Final Loss':>12} {'Accuracy':>10} {'Süre':>8}")
print("-" * 46)
for name, (loss, acc, elapsed) in results.items():
    print(f"{name:<12} {loss:>12.4f} {acc:>9.2f}% {elapsed:>7.1f}s")
