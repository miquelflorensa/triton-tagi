import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu") # TAGI is fast enough on CPU for this size, but works on GPU too.

# --- 1. Synthetic Data Generation ---
def create_sin_data(n=200):
    x = np.linspace(-6, 6, n)
    # Cubic function + noise (standard regression benchmark)
    y = x**3 + np.random.randn(n) * 3 
    return x, y

class Normalizer:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def denormalize(self, x):
        return x * self.std + self.mean

# --- 2. The Vectorized TAGI Layer ---
class TAGILayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # --- Initialize Parameters (He initialization) ---
        # We use nn.Parameter to register them, but we will update them manually.
        
        # He init: scale = sqrt(1 / fan_in)
        scale = np.sqrt(1.0 / in_features)
        
        # Mean of weights (mw) ~ N(0, scale)
        self.mw = nn.Parameter(torch.randn(in_features, out_features) * scale)
        # Variance of weights (Sw) = scale^2
        self.Sw = nn.Parameter(torch.full((in_features, out_features), scale ** 2))
        
        # Mean of biases (mb)
        self.mb = nn.Parameter(torch.zeros(1, out_features))
        # Variance of biases (Sb) = scale^2
        self.Sb = nn.Parameter(torch.full((1, out_features), scale ** 2))

    def forward(self, ma, Sa):
        """
        Forward propagation of moments (Mean and Variance).
        ma: Mean of activations from previous layer
        Sa: Variance of activations from previous layer
        """
        self.ma_in = ma  # Cache input mean for backward pass
        self.Sa_in = Sa  # Cache input var for backward pass
        
        # 1. Compute Mean of Z (Linear part)
        # mz = ma * mw + mb
        self.mz = torch.matmul(ma, self.mw) + self.mb
        
        # 2. Compute Variance of Z
        # Sz = (ma^2 * Sw) + (Sa * mw^2) + (Sa * Sw) + Sb
        term1 = torch.matmul(ma**2, self.Sw)
        term2 = torch.matmul(Sa, self.mw**2)
        term3 = torch.matmul(Sa, self.Sw)
        
        self.Sz = term1 + term2 + term3 + self.Sb
        
        return self.mz, self.Sz

    def backward(self, delta_mz, delta_Sz):
        """
        Analytical Parameter Update (Bayesian Inference).
        delta_mz: Normalized error signal for mean (from next layer)
        delta_Sz: Normalized error signal for variance (from next layer)
        """
        batch_size = delta_mz.shape[0]

        # --- 1. Calculate Gradients for Weights/Biases ---
        # Note: In TAGI, "gradient" refers to the update term Cov(param, z) * delta
        
        # Covariance between Z and W is approximately: input * Sw
        # Update mw: mw = mw + Sw * (input.T @ delta_mz)
        # We average over the batch
        
        # C_zw = self.ma_in.unsqueeze(2) * self.Sw.unsqueeze(0) # (Batch, In, Out)
        # Efficient formulation without expanding tensors:
        
        grad_mw = torch.matmul(self.ma_in.T, delta_mz) / batch_size
        grad_mb = torch.mean(delta_mz, dim=0, keepdim=True)
        
        # For variance, the update is multiplicative and based on squared input
        grad_Sw = torch.matmul(self.ma_in.T**2, delta_Sz) / batch_size
        grad_Sb = torch.mean(delta_Sz, dim=0, keepdim=True)

        # --- 2. Update Parameters ---
        # The update rule: param = param + gain * delta
        # Here, gain is effectively integrated into the grad calculation and Sw scaling
        
        self.mw.data += self.Sw * grad_mw
        self.mb.data += self.Sb * grad_mb
        
        # Variance update (ensure it stays positive)
        # Sw_new = Sw + Sw^2 * grad_Sw
        self.Sw.data += self.Sw**2 * grad_Sw
        self.Sw.data = torch.clamp(self.Sw.data, min=1e-6)
        
        self.Sb.data += self.Sb**2 * grad_Sb
        self.Sb.data = torch.clamp(self.Sb.data, min=1e-6)

        # --- 3. Compute Deltas for Previous Layer ---
        # We need to pass the error signal back to the previous layer
        
        # delta_ma = (delta_mz @ mw.T)
        delta_ma_next = torch.matmul(delta_mz, self.mw.T)
        
        # delta_Sa = (delta_Sz @ mw^2.T)
        delta_Sa_next = torch.matmul(delta_Sz, (self.mw**2).T)
        
        return delta_ma_next, delta_Sa_next

# --- 3. The TAGI Network ---
class TAGINet(nn.Module):
    def __init__(self, layers_struct):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_struct) - 1):
            self.layers.append(TAGILayer(layers_struct[i], layers_struct[i+1]))
            
    def forward(self, x):
        # Input has 0 variance (deterministic)
        ma = x
        Sa = torch.zeros_like(x)
        
        self.masks = [] # To store ReLU masks
        
        for i, layer in enumerate(self.layers):
            mz, Sz = layer(ma, Sa)
            
            # ReLU Activation (except last layer)
            if i < len(self.layers) - 1:
                # Mask where mean is positive
                mask = (mz > 0).float()
                self.masks.append(mask)
                
                # Propagate moments through ReLU
                ma = mz * mask
                Sa = Sz * mask # Simple approximation for variance
            else:
                # Output layer is linear
                ma = mz
                Sa = Sz
                self.masks.append(torch.ones_like(mz))
                
        return ma, Sa

    def tagi_step(self, x_batch, y_batch, sigma_v):
        """
        Performs one full step: Forward -> Loss Calc -> Backward Update
        """
        # 1. Forward
        y_pred_m, y_pred_S = self.forward(x_batch)
        
        # 2. Compute Innovation (Error) at Output
        # Total variance = Prediction Variance + Observation Noise
        Sy = y_pred_S + sigma_v**2
        
        # Delta Mean: (y - y_pred) / Sy
        delta_mz = (y_batch - y_pred_m) / Sy
        
        # Delta Variance: -1/Sy * (correction factor)
        # Standard TAGI update for variance reduces uncertainty based on observation
        delta_Sz = -1.0 / Sy 
        
        # 3. Backward Pass (Layer by Layer)
        for i in reversed(range(len(self.layers))):
            mask = self.masks[i]
            
            # Apply Activation Jacobian (Backprop through ReLU)
            # If mask is 0, error signal is killed
            dm = delta_mz * mask
            ds = delta_Sz * mask # Squared mask is same as mask for 0/1
            
            # Update layer and get deltas for previous
            delta_mz, delta_Sz = self.layers[i].backward(dm, ds)

# --- 4. Main Execution ---
def main():
    # Setup Data
    x_raw, y_raw = create_sin_data()
    
    # Shuffle and Split
    perm = np.random.permutation(len(x_raw))
    x_raw, y_raw = x_raw[perm], y_raw[perm]
    split = int(0.8 * len(x_raw))
    x_train, y_train = x_raw[:split], y_raw[:split]
    x_test, y_test = x_raw[split:], y_raw[split:]

    # Normalization
    x_norm = Normalizer(x_train)
    y_norm = Normalizer(y_train)
    
    x_tr_n = torch.tensor(x_norm.normalize(x_train), dtype=torch.float32).unsqueeze(1)
    y_tr_n = torch.tensor(y_norm.normalize(y_train), dtype=torch.float32).unsqueeze(1)
    
    # Hyperparameters
    net = TAGINet([1, 50, 1])
    sigma_v = 3.0 / y_norm.std # Observation noise in normalized space
    epochs = 50
    batch_size = 10
    
    # Training Loop
    print("Starting Training...")
    for epoch in range(epochs):
        # Shuffle batches
        perm = torch.randperm(x_tr_n.size(0))
        x_tr_n = x_tr_n[perm]
        y_tr_n = y_tr_n[perm]
        
        for i in range(0, len(x_tr_n), batch_size):
            xb = x_tr_n[i:i+batch_size]
            yb = y_tr_n[i:i+batch_size]
            
            # No gradients! pure analytical update
            with torch.no_grad():
                net.tagi_step(xb, yb, sigma_v)
                
    print("Training Complete.")

    # --- 5. Visualization ---
    # Generate smooth test curve
    x_vis = np.linspace(-7, 7, 300)
    x_vis_n = torch.tensor(x_norm.normalize(x_vis), dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        m_pred_n, S_pred_n = net(x_vis_n)
    
    # Denormalize
    y_pred = y_norm.denormalize(m_pred_n.numpy().flatten())
    # Std dev = sqrt(Var) * scale
    y_std = np.sqrt(S_pred_n.numpy().flatten()) * y_norm.std
    
    # Add observation noise to uncertainty for visual bounds
    # (Epistemic + Aleatoric uncertainty)
    y_std_total = np.sqrt(S_pred_n.numpy().flatten() + sigma_v**2) * y_norm.std

    # Plot
    plt.figure(figsize=(10, 6))
    plt.title("TAGI Regression (PyTorch Implementation)")
    
    # Plot Training Data
    plt.scatter(x_train, y_train, c='black', alpha=0.5, s=10, label='Train Data')
    
    # Plot Mean Prediction
    plt.plot(x_vis, y_pred, color='blue', linewidth=2, label='Mean Prediction')
    
    # Plot Uncertainty Bounds (Epistemic + Aleatoric)
    plt.fill_between(x_vis, 
                     y_pred - 2*y_std_total, 
                     y_pred + 2*y_std_total, 
                     color='blue', alpha=0.2, label='Uncertainty (±2$\sigma$)')
    
    # Plot Test Data
    plt.scatter(x_test, y_test, c='red', marker='x', s=20, label='Test Data')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    main()