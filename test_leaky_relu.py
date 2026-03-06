import torch
from src.layers import LeakyReLU

torch.manual_seed(42)
mz = torch.tensor([-1.0, 0.0, 1.0, 2.0], device="cuda")
Sz = torch.tensor([1.0, 2.0, 0.5, 1.0], device="cuda")

layer = LeakyReLU(alpha=0.1)

ma, Sa = layer.forward(mz, Sz)
print("ma:", ma)
print("Sa:", Sa)
print("J:", layer.J)

delta_mz = torch.ones_like(mz)
delta_Sz = torch.ones_like(Sz)

d_mz, d_Sz = layer.backward(delta_mz, delta_Sz)
print("d_mz:", d_mz)
print("d_Sz:", d_Sz)
