import numpy as np

with open("samples/Archer - 복사본/ArcherBodyPosition.buf", "rb") as f:
    pos_data = f.read()

pos = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 10)[:, :3].astype(np.float64)

with open("samples/Archer - 복사본/CustomPhysics/CustomPhysicsMap_0.buf", "rb") as f:
    map_data = f.read()

map_arr = np.frombuffer(map_data, dtype=np.float32).reshape(-1, 4)
wl = map_arr[:, 0]
wr = map_arr[:, 1]

non_phys = (wl < 0.01) & (wr < 0.01)

mask0 = wl > 0.05
if mask0.any():
    c0 = pos[mask0].mean(axis=0)
    d0 = np.linalg.norm(pos - c0, axis=1)
    d0[~non_phys] = np.inf
    root0 = int(d0.argmin())
    print(f"Slot 0 Anchor: idx={root0}, pos=({pos[root0,0]:.4f}, {pos[root0,1]:.4f}, {pos[root0,2]:.4f}), dist={d0[root0]:.4f}")

mask1 = wr > 0.05
if mask1.any():
    c1 = pos[mask1].mean(axis=0)
    d1 = np.linalg.norm(pos - c1, axis=1)
    d1[~non_phys] = np.inf
    root1 = int(d1.argmin())
    print(f"Slot 1 Anchor: idx={root1}, pos=({pos[root1,0]:.4f}, {pos[root1,1]:.4f}, {pos[root1,2]:.4f}), dist={d1[root1]:.4f}")
