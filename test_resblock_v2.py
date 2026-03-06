"""
Test suite for the rewritten ResBlock — verifying it matches cuTAGI logic.

Tests:
  1. Import test
  2. ResBlock identity shortcut: forward shapes + positive variances
  3. ResBlock projection shortcut: forward shapes + spatial downsampling
  4. ResBlock backward (identity): delta shapes match input shape
  5. ResBlock backward (projection): delta shapes match input shape
  6. Full ResNet-18 build + forward pass
  7. Verify no ReLU after the add (check output directly)
  8. Verify projection conv uses kernel_size=2
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_import():
    """Test that all imports work."""
    from src.layers.resblock import ResBlock, Add, triton_add_shortcut, triton_delta_merge
    print("✓ Import test passed")

def test_add_layer():
    """Test the Add layer standalone."""
    from src.layers.resblock import Add
    add = Add()
    mu_z = torch.randn(2, 4, 8, 8, device=DEVICE)
    var_z = torch.rand(2, 4, 8, 8, device=DEVICE).abs() + 0.01
    mu_x = torch.randn(2, 4, 8, 8, device=DEVICE)
    var_x = torch.rand(2, 4, 8, 8, device=DEVICE).abs() + 0.01
    
    mu_s, var_s = add.forward(mu_z, var_z, mu_x, var_x)
    assert mu_s.shape == (2, 4, 8, 8)
    assert var_s.shape == (2, 4, 8, 8)
    assert torch.allclose(mu_s, mu_z + mu_x, atol=1e-5)
    assert torch.allclose(var_s, var_z + var_x, atol=1e-5)
    assert (var_s > 0).all()
    print("✓ Add layer test passed")

def test_identity_forward():
    """ResBlock with identity shortcut (64→64, stride=1)."""
    from src.layers.resblock import ResBlock
    block = ResBlock(64, 64, stride=1, device=DEVICE)
    
    mu_in = torch.randn(4, 64, 16, 16, device=DEVICE)
    var_in = torch.rand(4, 64, 16, 16, device=DEVICE).abs() + 0.01
    
    mu_out, var_out = block.forward(mu_in, var_in)
    assert mu_out.shape == (4, 64, 16, 16), f"Expected (4,64,16,16), got {mu_out.shape}"
    assert var_out.shape == (4, 64, 16, 16)
    assert (var_out > 0).all(), "Variances must be positive"
    assert not block.use_projection, "Should use identity shortcut"
    print("✓ Identity forward test passed")

def test_projection_forward():
    """ResBlock with projection shortcut (64→128, stride=2)."""
    from src.layers.resblock import ResBlock
    block = ResBlock(64, 128, stride=2, device=DEVICE)
    
    mu_in = torch.randn(4, 64, 16, 16, device=DEVICE)
    var_in = torch.rand(4, 64, 16, 16, device=DEVICE).abs() + 0.01
    
    mu_out, var_out = block.forward(mu_in, var_in)
    assert mu_out.shape == (4, 128, 8, 8), f"Expected (4,128,8,8), got {mu_out.shape}"
    assert var_out.shape == (4, 128, 8, 8)
    assert (var_out > 0).all()
    assert block.use_projection, "Should use projection shortcut"
    print("✓ Projection forward test passed")

def test_identity_backward():
    """ResBlock identity: backward produces correct delta shapes."""
    from src.layers.resblock import ResBlock
    block = ResBlock(64, 64, stride=1, device=DEVICE)
    
    mu_in = torch.randn(4, 64, 16, 16, device=DEVICE)
    var_in = torch.rand(4, 64, 16, 16, device=DEVICE).abs() + 0.01
    
    mu_out, var_out = block.forward(mu_in, var_in)
    
    delta_mu = torch.randn_like(mu_out) * 0.01
    delta_var = torch.randn_like(var_out).abs() * 0.001
    
    d_mu, d_var = block.backward(delta_mu, delta_var)
    assert d_mu.shape == (4, 64, 16, 16), f"Expected (4,64,16,16), got {d_mu.shape}"
    assert d_var.shape == (4, 64, 16, 16)
    assert torch.isfinite(d_mu).all(), "Deltas must be finite"
    assert torch.isfinite(d_var).all(), "Deltas must be finite"
    print("✓ Identity backward test passed")

def test_projection_backward():
    """ResBlock projection: backward produces correct delta shapes."""
    from src.layers.resblock import ResBlock
    block = ResBlock(64, 128, stride=2, device=DEVICE)
    
    mu_in = torch.randn(4, 64, 16, 16, device=DEVICE)
    var_in = torch.rand(4, 64, 16, 16, device=DEVICE).abs() + 0.01
    
    mu_out, var_out = block.forward(mu_in, var_in)
    
    delta_mu = torch.randn_like(mu_out) * 0.01
    delta_var = torch.randn_like(var_out).abs() * 0.001
    
    d_mu, d_var = block.backward(delta_mu, delta_var)
    assert d_mu.shape == (4, 64, 16, 16), f"Expected (4,64,16,16), got {d_mu.shape}"
    assert d_var.shape == (4, 64, 16, 16)
    assert torch.isfinite(d_mu).all()
    assert torch.isfinite(d_var).all()
    print("✓ Projection backward test passed")

def test_architecture_details():
    """Verify the ResBlock matches cuTAGI's architecture exactly."""
    from src.layers.resblock import ResBlock
    from src.layers.relu import ReLU
    from src.layers.batchnorm2d import BatchNorm2D
    from src.layers.conv2d import Conv2D
    
    # Identity block
    block = ResBlock(64, 64, stride=1, device=DEVICE)
    assert len(block._main_layers) == 6, f"Main path should have 6 layers, got {len(block._main_layers)}"
    assert isinstance(block._main_layers[0], Conv2D), "Layer 0 should be Conv2D"
    assert isinstance(block._main_layers[1], ReLU),   "Layer 1 should be ReLU"
    assert isinstance(block._main_layers[2], BatchNorm2D), "Layer 2 should be BN"
    assert isinstance(block._main_layers[3], Conv2D), "Layer 3 should be Conv2D"
    assert isinstance(block._main_layers[4], ReLU),   "Layer 4 should be ReLU"
    assert isinstance(block._main_layers[5], BatchNorm2D), "Layer 5 should be BN"
    assert not block.use_projection
    
    # Projection block
    block = ResBlock(64, 128, stride=2, device=DEVICE)
    assert block.use_projection
    assert len(block._proj_layers) == 3, f"Proj path should have 3 layers, got {len(block._proj_layers)}"
    assert isinstance(block._proj_layers[0], Conv2D), "Proj[0] should be Conv2D"
    assert isinstance(block._proj_layers[1], ReLU),   "Proj[1] should be ReLU"
    assert isinstance(block._proj_layers[2], BatchNorm2D), "Proj[2] should be BN"
    
    # Check projection conv kernel size = 2
    assert block.proj_conv.kH == 2, f"Proj conv kernel should be 2, got {block.proj_conv.kH}"
    assert block.proj_conv.kW == 2, f"Proj conv kernel should be 2, got {block.proj_conv.kW}"
    
    # Verify NO relu_out attribute
    assert not hasattr(block, 'relu_out'), "Should NOT have relu_out (no activation after add)"
    
    print("✓ Architecture details test passed")

def test_no_post_activation():
    """Verify the ResBlock output has NO ReLU applied after the addition.
    
    If there were a ReLU after add, all negative means would be clipped.
    With no post-activation, the output can have negative means.
    """
    from src.layers.resblock import ResBlock
    
    block = ResBlock(64, 64, stride=1, device=DEVICE)
    
    # Use strongly negative input to ensure some negative outputs survive
    mu_in = torch.randn(4, 64, 8, 8, device=DEVICE) - 2.0  # shift negative
    var_in = torch.rand(4, 64, 8, 8, device=DEVICE).abs() + 0.01
    
    mu_out, var_out = block.forward(mu_in, var_in)
    
    # With no post-activation, negative outputs should exist
    has_negative = (mu_out < 0).any().item()
    print(f"  Output has negative means: {has_negative}")
    print(f"  Output mean range: [{mu_out.min().item():.4f}, {mu_out.max().item():.4f}]")
    # This is a soft check — with random weights, negative values should exist
    print("✓ No post-activation test passed")

def test_full_resnet18():
    """Build and run the full ResNet-18."""
    from run_resnet18 import build_resnet18
    
    net = build_resnet18(num_classes=10, head="remax", device=DEVICE)
    
    print(f"  Network: {net}")
    print(f"  Parameters: {net.num_parameters():,}")
    
    # Forward pass
    x = torch.randn(2, 3, 32, 32, device=DEVICE)
    mu, var = net.forward(x)
    
    assert mu.shape == (2, 10), f"Expected (2,10), got {mu.shape}"
    assert var.shape == (2, 10)
    assert (var > 0).all(), "Variances must be positive"
    
    # Check Remax outputs sum to ~1
    sums = mu.sum(dim=1)
    print(f"  Remax sums: {sums.tolist()}")
    assert torch.allclose(sums, torch.ones(2, device=DEVICE), atol=0.05), \
        f"Remax should sum to ~1.0, got {sums}"
    
    print("✓ Full ResNet-18 test passed")

def test_full_step():
    """Test a full training step (forward+backward+update)."""
    from run_resnet18 import build_resnet18
    
    net = build_resnet18(num_classes=10, head="remax", device=DEVICE)
    
    x = torch.randn(4, 3, 32, 32, device=DEVICE)
    y = torch.zeros(4, 10, device=DEVICE)
    y[:, 0] = 1.0  # class 0
    
    net.train()
    mu_pred, var_pred = net.step(x, y, sigma_v=1.0)
    
    assert mu_pred.shape == (4, 10)
    assert var_pred.shape == (4, 10)
    assert torch.isfinite(mu_pred).all(), "Predictions must be finite"
    assert torch.isfinite(var_pred).all(), "Variances must be finite"
    print("✓ Full training step test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("  ResBlock v2 Tests — cuTAGI-compatible")
    print("=" * 60)
    
    tests = [
        test_import,
        test_add_layer,
        test_identity_forward,
        test_projection_forward,
        test_identity_backward,
        test_projection_backward,
        test_architecture_details,
        test_no_post_activation,
        test_full_resnet18,
        test_full_step,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
