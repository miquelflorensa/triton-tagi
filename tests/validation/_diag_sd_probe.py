"""Probe what pytagi state_dict looks like before/after to_device('cuda')."""
import math
import torch
import pytagi
from pytagi.nn import AvgPool2d, BatchNorm2d, Conv2d, Linear, MixtureReLU, Remax, Sequential

torch.manual_seed(0); pytagi.manual_seed(0)
net = Sequential(
    Conv2d(3, 32, 5, padding=2, in_width=32, in_height=32),
    MixtureReLU(), BatchNorm2d(32), AvgPool2d(2, 2),
    Linear(8*8*32, 10), Remax(),
)
net.preinit_layer()
print("=== BEFORE to_device ===")
for k, v in net.state_dict().items():
    print(f"  {k:30s}  n_fields={len(v)}  len[0]={len(v[0]) if v and hasattr(v[0],'__len__') else 'scalar'}")

net.to_device("cuda")
print("\n=== AFTER to_device('cuda') ===")
for k, v in net.state_dict().items():
    print(f"  {k:30s}  n_fields={len(v)}  len[0]={len(v[0]) if v and hasattr(v[0],'__len__') else 'scalar'}")

# Try get_parameters if available
print("\n=== Methods on net ===")
for m in sorted(dir(net)):
    if not m.startswith('_'):
        print(f"  {m}")
