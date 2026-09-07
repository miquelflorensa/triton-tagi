# ImageNet frozen-feature AGCI pilot

This experiment freezes torchvision's pretrained ImageNet-1K ResNet-18,
caches its 512-dimensional penultimate features, and trains only a 1,000-way
AGCI/TAGI last layer. The default TAGI weight and bias gains are both 1.0.

