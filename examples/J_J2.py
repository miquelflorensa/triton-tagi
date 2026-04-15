import numpy as np

# ==========================================
# 1. Setup the Network State
# ==========================================
np.random.seed(42)
A = 1000               # Layer width
sigma_Z_sq = 1.0       # Base activation variance

# Initialize means (m_i) and uniform variances
mu_Z = np.random.normal(0, 1.0, A)
var_Z = np.full(A, sigma_Z_sq)

print(f"--- S CONSTRAINT (Linear) ---")
# Current aggregate variance
var_S = np.sum(var_Z)  # Should be 1000.0
target_var_S = 500.0   # We want to halve the aggregate variance

print(f"Original sum of marginal variances: {var_S:.1f}")
print(f"Target sum of marginal variances:   {target_var_S:.1f}")

# Method A: Standard Diagonal RTS (J^2)
J_S = var_Z / var_S
delta_var_S = target_var_S - var_S
var_Z_standard_RTS = var_Z + (J_S**2) * delta_var_S
sum_var_Z_standard_RTS = np.sum(var_Z_standard_RTS)

# Method B: Your Surrogate Moment-Matching Projection (J)
delta_sigma_S_sq = (target_var_S - var_S) / (var_S**2)
var_Z_proposed = var_Z + var_Z * (var_S * delta_sigma_S_sq)
sum_var_Z_proposed = np.sum(var_Z_proposed)

print(f"Post-Update (Standard RTS):         {sum_var_Z_standard_RTS:.4f}  <-- Fails due to 1/A^2 dilution")
print(f"Post-Update (Your Projection):      {sum_var_Z_proposed:.4f}  <-- Perfect match\n")


print(f"--- S2 CONSTRAINT (Quadratic) ---")
# Track two specific nodes to show the Jacobian weighting
node_near_zero = np.argmin(np.abs(mu_Z))  # Node with mu_Z ~ 0
node_large = np.argmax(np.abs(mu_Z))      # Node with large mu_Z

# Forward moments for Z^2 and S2
mu_Z2 = mu_Z**2 + var_Z
var_Z2 = 2 * var_Z**2 + 4 * var_Z * mu_Z**2
var_S2 = np.sum(var_Z2)

# Arbitrary target to trigger an update (reduce variance by 20%)
target_var_S2 = var_S2 * 0.8  

# Apply your S2 RTS update
delta_var_S2 = (target_var_S2 - var_S2) / (var_S2**2)
gain_numerator = 2 * mu_Z * var_Z
var_Z_updated_S2 = var_Z + (gain_numerator**2) * delta_var_S2

print(f"Update magnitude for node with mean ~ 0: {abs(var_Z_updated_S2[node_near_zero] - var_Z[node_near_zero]):.8f}")
print(f"Update magnitude for node with large mean: {abs(var_Z_updated_S2[node_large] - var_Z[node_large]):.8f}")
print(f"Ratio of updates (Large / Near-Zero):      {abs(var_Z_updated_S2[node_large] - var_Z[node_large]) / abs(var_Z_updated_S2[node_near_zero] - var_Z[node_near_zero] + 1e-12):.1f}x")