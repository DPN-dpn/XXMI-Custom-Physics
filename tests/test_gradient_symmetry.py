import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
from source.bone.weighting import calculate_anatomical_weights_with_anti_clipping

# Test 1: Symmetrical points
# Breast apex at tail (0.07, -0.14, 1.42), head at chest wall (0.07, -0.02, 1.42)
head_r = np.array([-0.07, -0.02, 1.42])
tail_r = np.array([-0.07, -0.14, 1.42])
radius = 0.08

head_l = np.array([0.07, -0.02, 1.42])
tail_l = np.array([0.07, -0.14, 1.42])

# Generate a grid of test points for Right side
xs_r = np.linspace(-0.13, -0.01, 20)
ys_r = np.linspace(-0.14, -0.02, 20)
zs_r = np.linspace(1.36, 1.48, 20)
grid_r = np.array([[x, y, z] for x in xs_r for y in ys_r for z in zs_r])

# Mirrored grid for Left side
grid_l = grid_r.copy()
grid_l[:, 0] = -grid_l[:, 0] # flip X

w_r = calculate_anatomical_weights_with_anti_clipping(grid_r, head_r, tail_r, radius, falloff_p=1.0, peak_w=0.40)
w_l = calculate_anatomical_weights_with_anti_clipping(grid_l, head_l, tail_l, radius, falloff_p=1.0, peak_w=0.40)

# Check max difference between left and right weights
diff = np.abs(w_r - w_l)
max_diff = np.max(diff)
print(f"Symmetry Test: Max difference between left and right weights = {max_diff:.8f}")
assert max_diff < 1e-6, f"Symmetry failed: max diff = {max_diff}"

# Test 2: Check gradient smoothness from head (chest wall) to tail (apex)
# Center line along depth: X = 0.07, Z = 1.42, Y from -0.02 (head) to -0.14 (tail)
center_depth_pts = np.array([[0.07, y, 1.42] for y in np.linspace(-0.02, -0.14, 13)])
w_depth = calculate_anatomical_weights_with_anti_clipping(center_depth_pts, head_l, tail_l, radius, falloff_p=1.0, peak_w=0.40)
print("Center Depth Gradient (Chest Wall -> Apex):")
for y, w in zip(np.linspace(-0.02, -0.14, 13), w_depth):
    print(f"  Y = {y:.4f}: weight = {w:.4f}")

# At chest wall (Y = -0.02), weight must be 0.0
assert w_depth[0] < 0.005, f"Chest wall weight not zero: {w_depth[0]}"
# At apex (Y = -0.14), weight must be peak_w (0.40)
assert abs(w_depth[-1] - 0.40) < 0.005, f"Apex weight not 0.40: {w_depth[-1]}"
# Gradient must be monotonically increasing from head to tail
diffs = np.diff(w_depth)
assert np.all(diffs >= -1e-6), "Depth gradient not monotonic!"

# Test 3: Check radial gradient smoothness from apex outward
center_radial_pts = np.array([[0.07 + dx, -0.14, 1.42] for dx in np.linspace(0.0, 0.08, 9)])
w_radial = calculate_anatomical_weights_with_anti_clipping(center_radial_pts, head_l, tail_l, radius, falloff_p=1.0, peak_w=0.40)
print("Radial Gradient (Apex -> Periphery):")
for dx, w in zip(np.linspace(0.0, 0.08, 9), w_radial):
    print(f"  Offset = {dx*100:.1f} cm: weight = {w:.4f}")

assert abs(w_radial[0] - 0.40) < 0.005, f"Apex weight not 0.40: {w_radial[0]}"
assert w_radial[-1] < 0.005, f"Periphery weight not zero: {w_radial[-1]}"
diffs_r = np.diff(w_radial)
assert np.all(diffs_r <= 1e-6), "Radial gradient not monotonic!"

print("ALL TESTS PASSED SUCCESSFULLY!")
