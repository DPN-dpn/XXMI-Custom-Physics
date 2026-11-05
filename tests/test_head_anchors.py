import numpy as np
import json

with open("samples/Archer - 복사본/ArcherBodyPosition.buf", "rb") as f:
    pos_data = f.read()

buf_np = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 10)[:, :3].astype(np.float64)

with open("samples/Archer - 복사본/CustomPhysics/CustomPhysicsMap_0.buf", "rb") as f:
    map_data = f.read()

map_arr = np.frombuffer(map_data, dtype=np.float32).reshape(-1, 4)
wl = map_arr[:, 0]
wr = map_arr[:, 1]
non_phys = (wl < 0.01) & (wr < 0.01)

with open("samples/Archer - 복사본/CustomPhysics/XXMI_Physics_Config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

bones = cfg.get("bones", {})
for bname, bdata in bones.items():
    head_blender = np.array(bdata["head"])
    # Blender -> DX coordinates: (x, z, -y)
    head_dx = np.array([head_blender[0], head_blender[2], -head_blender[1]])
    
    dists = np.linalg.norm(buf_np - head_dx, axis=1)
    dists[~non_phys] = np.inf
    best_idx = int(dists.argmin())
    best_p = buf_np[best_idx]
    print(f"Bone '{bname}': Head_DX=({head_dx[0]:.4f}, {head_dx[1]:.4f}, {head_dx[2]:.4f}) -> Anchor idx={best_idx}, pos=({best_p[0]:.4f}, {best_p[1]:.4f}, {best_p[2]:.4f}), dist={dists[best_idx]:.4f}")
