import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0 bekleniyor
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0 bekleniyor
print("\nKendi Value sınıfımızın verdiği sonuçla birebir aynı olmalı.")
