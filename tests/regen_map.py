import sys, os, struct, json
sys.path.insert(0, os.getcwd())
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree
from source.bone.weighting import calculate_anatomical_weights_with_anti_clipping

pos_path = "samples/Archer - 복사본/ArcherBodyPosition.buf"
with open(pos_path, "rb") as f:
    pos_data = f.read()

stride = 40
n_buf = len(pos_data) // stride
buf_np = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 10)[:, :3].astype(np.float64)
buf_norm_sr = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 10)[:, 3:6].astype(np.float64)

# In Blender coords: X=x, Y=z, Z=y
pts_blender = np.column_stack([buf_np[:, 0], -buf_np[:, 2], buf_np[:, 1]])
buf_norm_blender = np.column_stack([buf_norm_sr[:, 0], -buf_norm_sr[:, 2], buf_norm_sr[:, 1]])

with open("samples/Archer - 복사본/CustomPhysics/XXMI_Physics_Config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

bones = cfg.get("bones", {})
keys = list(bones.keys())
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
phys_idx = np.nonzero((wl_out > 0.005) | (wr_out > 0.005))[0]
kd = KDTree(len(phys_idx))
for k, bi in enumerate(phys_idx):
    kd.insert(Vector(pts_blender[bi]), k)
kd.balance()

for w_arr in (wl_out, wr_out):
    orig_w = w_arr.copy()
    promotions = {}
    clamps = {}
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
            if float(cur_n @ hit_n) < 0.50: continue
            delta = hit_p - cur_p
            d_norm = float(delta @ cur_n)
            abs_d_norm = abs(d_norm)
            if abs_d_norm < 0.0008 or abs_d_norm > 0.015: continue
            d_perp_sq = float(delta @ delta) - d_norm * d_norm
            if d_perp_sq > (0.0040 * 0.0040): continue
            
            w_hit = float(orig_w[hit_bi])
            if d_norm > 0:
                if w_hit < cur_w:
                    promotions[hit_bi] = max(promotions.get(hit_bi, w_hit), cur_w)
            else:
                if cur_w < w_hit:
                    clamps[bi] = min(clamps.get(bi, cur_w), w_hit)

    for bi, val in promotions.items():
        w_arr[bi] = max(w_arr[bi], val)
    for bi, val in clamps.items():
        w_arr[bi] = min(w_arr[bi], val)

# Boundary gradient taper using index buffer
ib_path = "samples/Archer - 복사본/ArcherBodyB.ib"
if os.path.exists(ib_path):
    with open(ib_path, "rb") as f:
        ib_data = f.read()
    ib = np.frombuffer(ib_data, dtype=np.uint32)
    if len(ib) == 0 or ib.max() >= n_buf:
        ib = np.frombuffer(ib_data, dtype=np.uint16)
    triangles = ib.reshape(-1, 3)
    from collections import defaultdict
    adj = defaultdict(set)
    for t in triangles:
        adj[t[0]].add(t[1]); adj[t[0]].add(t[2])
        adj[t[1]].add(t[0]); adj[t[1]].add(t[2])
        adj[t[2]].add(t[0]); adj[t[2]].add(t[1])

    for w_arr in (wl_out, wr_out):
        for _ in range(3):
            active_b = np.nonzero(w_arr > 0.01)[0]
            for vi in active_b:
                nbrs = adj[vi]
                if not nbrs: continue
                min_nb = min(w_arr[nb] for nb in nbrs)
                diff = w_arr[vi] - min_nb
                if diff > 0.08:
                    w_arr[vi] -= (diff - 0.08) * 0.45

# Coincident vertex synchronization across all identical coordinates (DSU)
kd_co = KDTree(n_buf)
for vi, p in enumerate(buf_np):
    kd_co.insert(Vector(p), vi)
kd_co.balance()

parent = list(range(n_buf))
def find_p(i):
    path = []
    while parent[i] != i:
        path.append(i)
        i = parent[i]
    for node in path:
        parent[node] = i
    return i
def union_p(i, j):
    pi, pj = find_p(i), find_p(j)
    if pi != pj: parent[pi] = pj

active_buf_idx = np.nonzero((wl_out > 0.0001) | (wr_out > 0.0001))[0]
for vi in active_buf_idx:
    hits = kd_co.find_range(Vector(buf_np[vi]), 0.0008)
    for _, hit_k, _ in hits:
        union_p(vi, hit_k)

co_groups_map = {}
for vi in range(n_buf):
    root = find_p(vi)
    co_groups_map.setdefault(root, []).append(vi)

for root, g_list in co_groups_map.items():
    if len(g_list) > 1 and any(wl_out[v] > 0.0001 or wr_out[v] > 0.0001 for v in g_list):
        max_l = max(float(wl_out[v]) for v in g_list)
        max_r = max(float(wr_out[v]) for v in g_list)
        for v in g_list:
            wl_out[v] = max_l
            wr_out[v] = max_r

map_bytes = np.column_stack([wl_out, wr_out, np.zeros(n_buf, dtype=np.float32), np.zeros(n_buf, dtype=np.float32)]).astype("<f4").tobytes()
map_path = "samples/Archer - 복사본/CustomPhysics/CustomPhysicsMap_0.buf"
with open(map_path, "wb") as f:
    f.write(map_bytes)

print("Regenerated CustomPhysicsMap_0.buf with smooth gradient, non-penetration and 0 coincident mismatch!")
print("wl > 0.001:", int(np.count_nonzero(wl_out > 0.001)), "max:", float(wl_out.max()), "mean:", float(wl_out[wl_out > 0.001].mean()))
print("wr > 0.001:", int(np.count_nonzero(wr_out > 0.001)), "max:", float(wr_out.max()), "mean:", float(wr_out[wr_out > 0.001].mean()))
