import sys, os, struct, json
sys.path.insert(0, os.getcwd())
import numpy as np
from source.bone.weighting import calculate_anatomical_weights_with_anti_clipping

pos_path = "samples/Archer - 복사본/ArcherBodyPosition.buf"
with open(pos_path, "rb") as f:
    pos_data = f.read()

stride = 40
n_buf = len(pos_data) // stride
buf_np = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 10)[:, :3].astype(np.float64)

# In Blender coords: X=x, Y=z, Z=y
# In Star Rail coords: X=x, Y=y (height), Z=z (depth)
# Transform Star Rail (x, y, z) to Blender: blender_x = sr_x, blender_y = -sr_z, blender_z = sr_y
pts_blender = np.column_stack([buf_np[:, 0], -buf_np[:, 2], buf_np[:, 1]])

with open("samples/Archer - 복사본/CustomPhysics/XXMI_Physics_Config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

bones = cfg.get("bones", {})
keys = list(bones.keys()) # right, left
b_r = bones[keys[0]] if "오" in keys[0] or ".r" in keys[0].lower() else bones[keys[1]]
b_l = bones[keys[1]] if "오" in keys[0] or ".r" in keys[0].lower() else bones[keys[0]]

head_r = np.array(b_r["head"], dtype=np.float64)
tail_r = np.array(b_r["tail"], dtype=np.float64)
radius_r = float(b_r["radius"])

head_l = np.array(b_l["head"], dtype=np.float64)
tail_l = np.array(b_l["tail"], dtype=np.float64)
radius_l = float(b_l["radius"])

print("Head R:", head_r, "Tail R:", tail_r, "Radius:", radius_r)
print("Head L:", head_l, "Tail L:", tail_l, "Radius:", radius_l)

wr_out = calculate_anatomical_weights_with_anti_clipping(pts_blender, head_r, tail_r, radius_r, falloff_p=1.0, peak_w=0.40).astype(np.float32)
wl_out = calculate_anatomical_weights_with_anti_clipping(pts_blender, head_l, tail_l, radius_l, falloff_p=1.0, peak_w=0.40).astype(np.float32)

# Midline cross blend
sum_w = wl_out + wr_out
over = (wl_out > 0.001) & (wr_out > 0.001) & (sum_w > 0.40)
if np.any(over):
    scale = 0.40 / np.maximum(1e-5, sum_w[over])
    wl_out[over] *= scale
    wr_out[over] *= scale

# Multi-layer synchronization across layers (skin vs cloth separated by normal)
buf_norm_sr = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 10)[:, 3:6].astype(np.float64)
buf_norm_blender = np.column_stack([buf_norm_sr[:, 0], -buf_norm_sr[:, 2], buf_norm_sr[:, 1]])

from mathutils.kdtree import KDTree
from mathutils import Vector

phys_idx = np.nonzero((wl_out > 0.005) | (wr_out > 0.005))[0]
kd = KDTree(len(phys_idx))
for k, bi in enumerate(phys_idx):
    kd.insert(Vector(pts_blender[bi]), k)
kd.balance()

for w_arr in (wl_out, wr_out):
    orig_w = w_arr.copy()
    updates = {}
    for k, bi in enumerate(phys_idx):
        cur_w = float(orig_w[bi])
        if cur_w <= 0.005: continue
        cur_p = pts_blender[bi]
        cur_n = buf_norm_blender[bi]
        hits = kd.find_range(Vector(cur_p), 0.015)
        for _, hit_k, _ in hits:
            if hit_k == k: continue
            hit_bi = phys_idx[hit_k]
            hit_p = pts_blender[hit_bi]
            hit_n = buf_norm_blender[hit_bi]
            dot_n = float(cur_n @ hit_n)
            if dot_n < 0.60: continue
            delta = hit_p - cur_p
            d_norm = float(delta @ cur_n)
            abs_d_norm = abs(d_norm)
            if abs_d_norm < 0.0008 or abs_d_norm > 0.015: continue
            d2 = float(delta @ delta)
            d_perp_sq = max(0.0, d2 - d_norm * d_norm)
            if d_perp_sq > (0.0050 * 0.0050): continue
            unified = max(cur_w, float(orig_w[hit_bi]))
            if unified > updates.get(bi, 0.0): updates[bi] = unified
            if unified > updates.get(hit_bi, 0.0): updates[hit_bi] = unified
    for bi, uw in updates.items():
        w_arr[bi] = uw

map_bytes = np.column_stack([wl_out, wr_out, np.zeros(n_buf, dtype=np.float32), np.zeros(n_buf, dtype=np.float32)]).astype("<f4").tobytes()
map_path = "samples/Archer - 복사본/CustomPhysics/CustomPhysicsMap_0.buf"
with open(map_path, "wb") as f:
    f.write(map_bytes)

print("Regenerated CustomPhysicsMap_0.buf with smooth gradient and non-cascading multi-layer sync!")
print("wl > 0.001:", int(np.count_nonzero(wl_out > 0.001)), "max:", float(wl_out.max()), "mean:", float(wl_out[wl_out > 0.001].mean()))
print("wr > 0.001:", int(np.count_nonzero(wr_out > 0.001)), "max:", float(wr_out.max()), "mean:", float(wr_out[wr_out > 0.001].mean()))

