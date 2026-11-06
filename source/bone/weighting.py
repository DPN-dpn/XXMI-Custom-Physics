import math
import numpy as np
from collections import deque
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import (
    ensure_object_visible,
    ensure_all_objects_in_object_mode,
    read_vertex_group_arrays,
    world_vertex_positions,
    mesh_edge_arrays,
    label_components,
    ROLE_RIGID_VG,
)
from .rig import (
    BREAST_RIG_NAME,
    VG_BREAST_L,
    VG_BREAST_R,
    get_rig_bone_info,
)
from ..rigid.data import get_rigid_data
from .data import get_bone_params, get_bone_item, get_bone_data

VG_RIGID = ROLE_RIGID_VG


def find_connected_islands(vert_indices, edges_list):
    """선택된 정점들 사이의 연결성(Mesh Islands)을 분석하여 독립된 덩어리들을 반환합니다."""
    adj = {v: [] for v in vert_indices}
    v_set = set(vert_indices)

    for v1, v2 in edges_list:
        if v1 in v_set and v2 in v_set:
            adj[v1].append(v2)
            adj[v2].append(v1)

    visited = set()
    islands = []

    for v in vert_indices:
        if v not in visited:
            island = []
            queue = deque([v])
            visited.add(v)
            while queue:
                curr = queue.popleft()
                island.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            islands.append(island)

    return islands


def calculate_anatomical_weights_with_anti_clipping(world_pts, head, tail_or_fwd, eff_radius, falloff_p=1.0, peak_w=0.80):
    """
    해부학적 앵커 본 세그먼트[Head -> Tail] 기반 3D 이중 감쇠(전방 깊이 Smoothstep + 방사형 Hann Cosine):
    1. 전방 깊이 감쇠: 흉벽(0.0)에서 꼭짓점(1.0)으로 매끄럽게 증가
    2. 방사형 감쇠: 중심축(0.0)에서 외곽 둘레(1.0)로 매끄럽게 감소하는 Raised Cosine(Hann) 곡선 적용
    3. 최대 가중치(peak_w): 일체형 의상에서 경계선이 과도하게 꺾이는 것을 방지하기 위해 최대치를 부드럽게 완화
    """
    if len(world_pts) == 0:
        return np.array([], dtype=np.float64)

    head_arr = np.array(head, dtype=np.float64)
    tail_arr = np.array(tail_or_fwd, dtype=np.float64)
    norm3 = float(np.linalg.norm(tail_arr))

    if abs(norm3 - 1.0) < 1e-3:
        fwd_u = tail_arr / norm3
        tail_arr = head_arr + fwd_u * float(eff_radius)

    bone_vec = tail_arr - head_arr
    bone_len = float(np.linalg.norm(bone_vec))
    if bone_len < 1e-6:
        bone_u = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        bone_len = max(1e-4, float(eff_radius))
    else:
        bone_u = bone_vec / bone_len

    pts_rel = world_pts - head_arr

    # 1. 전방 깊이 축 투영 거리 및 매끄러운 돔 감쇠 (C1 연속 코사인 램프: 흉벽에서 0.0, 꼭짓점에서 1.0)
    t_proj = pts_rel @ bone_u
    t_clamped = np.clip(t_proj / bone_len, 0.0, 1.0)
    w_depth = 0.5 * (1.0 - np.cos(np.pi * t_clamped))
    w_depth = np.where(t_proj > 0.0, w_depth, 0.0)

    # 2. 본 세그먼트 [Head -> Tail] 와의 3D 수직 최단거리 및 매끄러운 감쇠 곡선
    proj_pts = head_arr + np.outer(t_clamped, bone_vec)
    d_perp = np.linalg.norm(world_pts - proj_pts, axis=1)

    # 반경 마진 확장: 경계선 부근(가슴골, 옷깃, 외곽)에서 가중치가 0으로 급격히 절단되어
    # 뾰족한 스파이크 버그나 메쉬 찢어짐이 발생하는 것을 방지하기 위해 1.30배까지 매끄러운 코사인 감쇠 적용
    radius = max(0.02, float(eff_radius))
    max_radius = radius * 1.30
    rho_norm = np.clip(d_perp / max_radius, 0.0, 1.0)
    w_radial = np.where(d_perp < max_radius, 0.5 * (1.0 + np.cos(np.pi * rho_norm)), 0.0)

    # 3. 해부학적 합성 가중치 계산 (기저 지수 0.35 보정: 기본값 1.0에서 자연스럽고 풍성한 볼륨 형성)
    eff_power = 0.35 * max(0.05, float(falloff_p))
    w_comb = peak_w * np.power(np.maximum(0.0, w_depth * w_radial), eff_power)
    w_arr = np.where((w_radial > 0.0) & (w_depth > 0.0), np.clip(w_comb, 0.001, 1.0), 0.0)

    # 4. UV 심(Seam) 및 분리 정점 동기화: 0.5mm 이내 동일 3D 좌표 100% 동일 가중치 부여
    # (외곽 감쇠 그라데이션 보존: 동일 표면 내 수평 번짐 없이 순수 해부학적 Hann Cosine 감쇠 유지)
    if len(world_pts) > 1:
        kd = KDTree(len(world_pts))
        for k, pt in enumerate(world_pts.tolist()):
            kd.insert(Vector(pt), k)
        kd.balance()

        for idx in range(len(world_pts)):
            co = Vector(world_pts[idx])
            hits = kd.find_range(co, 0.0005)
            if len(hits) > 1:
                max_w = max(w_arr[h_idx] for _, h_idx, _ in hits)
                for _, h_idx, _ in hits:
                    w_arr[h_idx] = max_w

    return w_arr


def eval_bone_weight_at_point(pt, bname, bone_info, props=None, falloff_p=1.0, peak_w=0.40):
    """
    3D 월드 상의 한 점(강체의 중심점 등)에서 해당 물리 본(bname)에 의한
    해부학적 가중치를 평가하여 반환합니다.
    """
    if not bone_info:
        return 0.0

    entry = bone_info.get(bname)
    if not entry:
        if bname in ("L", "R"):
            entry = bone_info.get(bname)
        elif f"Breast.{bname}" in bone_info:
            entry = bone_info.get(f"Breast.{bname}")

    if not entry:
        return 0.0

    head = Vector(entry["head"])
    tail = Vector(entry["tail"])
    bone_vec = tail - head
    bone_len = bone_vec.length
    fwd_u = bone_vec.normalized() if bone_len > 1e-4 else Vector((0.0, -1.0, 0.0))
    radius = max(0.02, entry["radius"])

    v = pt - head
    t = v.dot(fwd_u)
    t_norm = max(0.0, min(1.0, t / max(1e-4, bone_len)))
    proj_pt = head + bone_vec * t_norm
    d_perp = (pt - proj_pt).length

    if d_perp >= radius:
        return 0.0

    rho_norm = d_perp / radius
    w_radial = 0.5 * (1.0 + math.cos(math.pi * rho_norm))
    w_depth = 0.5 * (1.0 - math.cos(math.pi * t_norm)) if t > 0.0 else 0.0

    if props:
        b_params = get_bone_params(props, bname)
        peak_w = b_params["peak_weight"]
        falloff_p = b_params["falloff_power"]

    eff_power = 0.35 * max(0.05, float(falloff_p))
    w = peak_w * math.pow(max(0.0, w_depth * w_radial), eff_power)
    return max(0.001, min(1.0, w))


def unify_rigid_clusters_across_objects(target_objs, bone_info, falloff_p, peak_w=0.80, exclude_verts_by_obj=None):
    """
    여러 오브젝트에 나뉘어 있는 강체(XXMI_Role_Rigid)들을
    3D 월드 공간에서 서로 맞닿아 있는(5mm 이내) 덩어리끼리 하나의 '통합 강체 클러스터'로 묶어
    완벽히 동일한 단일 (w_l, w_r) 가중치를 부여합니다.
    """
    if not target_objs or not bone_info:
        return 0

    if exclude_verts_by_obj is None:
        exclude_verts_by_obj = {}

    islands = []

    for obj in target_objs:
        if obj.type != 'MESH':
            continue
        vg_rigid = obj.vertex_groups.get(VG_RIGID)
        if vg_rigid is None:
            continue

        mesh = obj.data
        n_verts = len(mesh.vertices)
        rigid_mask = read_vertex_group_arrays(obj, {vg_rigid.index: "rigid"})["rigid"] > 0.5

        ex_verts = exclude_verts_by_obj.get(obj.name)
        if ex_verts:
            for ev in ex_verts:
                if ev < n_verts:
                    rigid_mask[ev] = False

        if not rigid_mask.any():
            continue

        e0, e1 = mesh_edge_arrays(mesh)
        lab_all = label_components(n_verts, e0, e1)
        uniq_labs, lab_counts = np.unique(lab_all, return_counts=True)
        for lab_id, cnt in zip(uniq_labs, lab_counts):
            if cnt <= 2500:
                isl_mask = (lab_all == lab_id)
                n_rig = np.count_nonzero(rigid_mask & isl_mask)
                if n_rig >= 5 or (n_rig / cnt) >= 0.15:
                    rigid_mask[isl_mask] = True
                    vg_rigid.add(np.nonzero(isl_mask)[0].tolist(), 1.0, 'REPLACE')

        world = world_vertex_positions(obj)
        sub = rigid_mask[e0] & rigid_mask[e1]
        lab = label_components(n_verts, e0[sub], e1[sub])

        ridx = np.nonzero(rigid_mask)[0]
        _, inv = np.unique(lab[ridx], return_inverse=True)
        order = np.argsort(inv, kind='stable')
        sorted_idx = ridx[order]
        splits = np.cumsum(np.bincount(inv))[:-1]
        for group in np.split(sorted_idx, splits):
            if len(group) == 0:
                continue
            pts = world[group]
            islands.append({
                "obj": obj,
                "verts": group.tolist(),
                "pts": pts,
                "bmin": pts.min(axis=0),
                "bmax": pts.max(axis=0),
                "cent": pts.mean(axis=0),
            })

    if not islands:
        return 0

    n_isl = len(islands)
    margin = 0.025
    parent = list(range(n_isl))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kd_cache = {}

    def _get_kd(i):
        kd = kd_cache.get(i)
        if kd is None:
            pts = islands[i]["pts"]
            kd = KDTree(len(pts))
            for k, p in enumerate(pts.tolist()):
                kd.insert(Vector(p), k)
            kd.balance()
            kd_cache[i] = kd
        return kd

    for i in range(n_isl):
        a = islands[i]
        for j in range(i + 1, n_isl):
            if _find(i) == _find(j):
                continue
            b = islands[j]
            if np.any(a["bmin"] - margin > b["bmax"]) or np.any(a["bmax"] + margin < b["bmin"]):
                continue

            connected = False
            cent_dist = float(np.linalg.norm(a["cent"] - b["cent"]))
            if cent_dist <= 0.06:
                connected = True

            if not connected:
                if (np.all(a["bmin"] - 0.015 <= b["cent"]) and np.all(b["cent"] <= a["bmax"] + 0.015)) or \
                   (np.all(b["bmin"] - 0.015 <= a["cent"]) and np.all(a["cent"] <= b["bmax"] + 0.015)):
                    connected = True

            if not connected:
                step = max(1, len(a["pts"]) // 40)
                kd_b = _get_kd(j)
                for p in a["pts"][::step].tolist():
                    if kd_b.find(Vector(p))[2] <= margin:
                        connected = True
                        break

            if connected:
                parent[_find(i)] = _find(j)

    clusters = {}
    for i in range(n_isl):
        clusters.setdefault(_find(i), []).append(islands[i])

    active_bones = [k for k in bone_info.keys() if k not in ("L", "R")]
    if not active_bones:
        active_bones = list(bone_info.keys())

    for comp in clusters.values():
        all_pts = np.concatenate([isl["pts"] for isl in comp], axis=0)
        c_cent = Vector(all_pts.mean(axis=0).tolist())

        best_bname = None
        best_dist = 1e9
        for bn in active_bones:
            bentry = bone_info.get(bn)
            if bentry:
                d = (c_cent - Vector(bentry["centroid"])).length
                if d < best_dist:
                    best_dist = d
                    best_bname = bn

        if not best_bname:
            continue

        c_w = eval_bone_weight_at_point(c_cent, best_bname, bone_info, falloff_p=falloff_p, peak_w=peak_w)
        vg_name = f"XXMI_{best_bname}"

        for isl in comp:
            obj = isl["obj"]
            verts = isl["verts"]
            vg = obj.vertex_groups.get(vg_name) or obj.vertex_groups.new(name=vg_name)
            if c_w > 0.0:
                vg.add(verts, c_w, 'REPLACE')
            else:
                vg.remove(verts)

    return len(clusters)


def smooth_mesh_weights(obj, vg_list, rigid_mask, iters=2, alpha=0.45, lock_registered=True, allowed_mask=None):
    """
    메쉬 엣지 위상 구조를 기반으로 지정된 버텍스 그룹들의 가중치를 라플라시안 스무딩합니다.
    """
    if iters <= 0 or not obj or obj.type != 'MESH' or not vg_list:
        return
    mesh = obj.data
    e0, e1 = mesh_edge_arrays(mesh)
    if len(e0) == 0:
        return
    n_verts = len(mesh.vertices)

    for vg in vg_list:
        if not vg:
            continue
        arrs = read_vertex_group_arrays(obj, {vg.index: "w"})
        w = arrs.get("w", np.zeros(n_verts, dtype=np.float32)).astype(np.float32)
        active_initial = (w > 0.0001) & ~rigid_mask
        if not np.any(active_initial):
            continue

        domain_mask = allowed_mask if (allowed_mask is not None) else active_initial

        for _ in range(iters):
            n_sum = np.zeros(n_verts, dtype=np.float32)
            n_cnt = np.zeros(n_verts, dtype=np.int32)
            np.add.at(n_sum, e0, w[e1])
            np.add.at(n_sum, e1, w[e0])
            np.add.at(n_cnt, e0, 1)
            np.add.at(n_cnt, e1, 1)
            has_n = n_cnt > 0
            avg_w = np.where(has_n, n_sum / np.maximum(1, n_cnt), w)
            w = np.where(rigid_mask, w, (1.0 - alpha) * w + alpha * avg_w)
            if lock_registered:
                w[~domain_mask & ~rigid_mask] = 0.0

        changed_idx = np.nonzero((w > 0.0001) & ~rigid_mask)[0]
        for vi in changed_idx.tolist():
            wi = float(w[vi])
            if wi > 0.0005:
                vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')
            else:
                vg.remove([vi])


def normalize_overlapping_bone_weights(target_objs, active_bones, max_total_weight=0.80):
    """
    여러 모핑 본(예: Left, Right)의 영향 영역이 겹치는 가슴 중앙/클리비지 정점에서
    가중치 합이 max_total_weight를 초과하여 변위가 2배로 폭발하거나 뾰족하게 튀어나오는 현상을 방지합니다.
    양쪽 본의 비율(L:R)은 정확히 보존하면서 합이 max_total_weight 이내가 되도록 부드럽게 크로스 블렌딩합니다.
    """
    if not target_objs or len(active_bones) < 2:
        return 0

    valid_objs = [o for o in target_objs if o and o.type == 'MESH']
    total_clamped = 0

    for obj in valid_objs:
        n_verts = len(obj.data.vertices)
        if n_verts == 0:
            continue

        # 활성 본들의 버텍스 그룹 및 가중치 배열 수집
        vg_dict = {}
        for bn in active_bones:
            vg = obj.vertex_groups.get(f"XXMI_{bn}")
            if not vg:
                if bn in ("L", "Breast.L"):
                    vg = obj.vertex_groups.get(VG_BREAST_L)
                elif bn in ("R", "Breast.R"):
                    vg = obj.vertex_groups.get(VG_BREAST_R)
            if vg:
                vg_dict[bn] = vg

        if len(vg_dict) < 2:
            continue

        gmap = {vg.index: bn for bn, vg in vg_dict.items()}
        arrs = read_vertex_group_arrays(obj, gmap)

        weight_matrix = np.zeros((len(vg_dict), n_verts), dtype=np.float32)
        bn_keys = list(vg_dict.keys())
        for row_idx, bn in enumerate(bn_keys):
            weight_matrix[row_idx] = arrs.get(bn, np.zeros(n_verts, dtype=np.float32))

        # 활성 가중치 합 계산
        sum_w = weight_matrix.sum(axis=0)

        vg_rigid = obj.vertex_groups.get(VG_RIGID)
        rigid_mask = np.zeros(n_verts, dtype=bool)
        if vg_rigid:
            r_data = read_vertex_group_arrays(obj, {vg_rigid.index: "r"}).get("r")
            if r_data is not None:
                rigid_mask = (r_data > 0.5)

        # 2개 이상의 본에 영향받으며, 합이 max_total_weight를 초과하는 비강체 정점 식별
        active_counts = (weight_matrix > 0.001).sum(axis=0)
        over_mask = (active_counts >= 2) & (sum_w > max_total_weight) & ~rigid_mask

        if not np.any(over_mask):
            continue

        over_idx = np.nonzero(over_mask)[0]
        for vi in over_idx.tolist():
            s = float(sum_w[vi])
            scale_factor = max_total_weight / s
            for row_idx, bn in enumerate(bn_keys):
                w_orig = float(weight_matrix[row_idx, vi])
                if w_orig > 0.0001:
                    new_w = w_orig * scale_factor
                    vg_dict[bn].add([vi], max(0.0005, min(1.0, new_w)), 'REPLACE')
            total_clamped += 1

    return total_clamped


def synchronize_coincident_vertices_across_objects(target_objs, vg_names=None, tol=0.0002):
    """
    모든 대상 오브젝트에 걸쳐 3D 공간 상에서 동일한 위치(기본 0.2mm 이내)를 공유하는
    모든 정점(UV 심, 하드 엣지 분리점 등)의 가중치를 100% 동일하게 동기화합니다.
    동적으로 등록된 모든 모핑 본 버텍스 그룹(XXMI_*)을 지원합니다.
    """
    if not target_objs:
        return 0

    valid_objs = [o for o in target_objs if o and o.type == 'MESH' and len(o.data.vertices) > 0]
    if not valid_objs:
        return 0

    if vg_names is None:
        vg_set = set()
        for obj in valid_objs:
            for vg in obj.vertex_groups:
                if vg.name.startswith("XXMI_") and vg.name not in (VG_RIGID, "XXMI_Role_Static", "XXMI_Physics"):
                    vg_set.add(vg.name)
        vg_names = sorted(list(vg_set))

    if not vg_names:
        return 0

    synced_count = 0

    for vgroup_name in vg_names:
        active_items = []
        obj_entries = []

        for o_idx, obj in enumerate(valid_objs):
            vg = obj.vertex_groups.get(vgroup_name)
            vg_rigid = obj.vertex_groups.get(VG_RIGID)

            gmap = {}
            if vg: gmap[vg.index] = "w"
            if vg_rigid: gmap[vg_rigid.index] = "rigid"

            arrs = read_vertex_group_arrays(obj, gmap)
            world = world_vertex_positions(obj)
            n_verts = len(world)

            w = arrs.get("w", np.zeros(n_verts, dtype=np.float32))
            rigid = (arrs.get("rigid", np.zeros(n_verts, dtype=np.float32)) > 0.5)

            obj_entries.append({
                "obj": obj,
                "vg": vg,
                "world": world,
                "w": w,
                "rigid": rigid,
                "n": n_verts,
            })

            mask = (w > 0.0001) & ~rigid
            for vi in np.nonzero(mask)[0].tolist():
                active_items.append((o_idx, vi, world[vi], float(w[vi])))

        if not active_items:
            continue

        kd = KDTree(len(active_items))
        for k, item in enumerate(active_items):
            kd.insert(Vector(item[2]), k)
        kd.balance()

        updates_per_obj = {o_idx: {} for o_idx in range(len(obj_entries))}

        for o_idx, entry in enumerate(obj_entries):
            world = entry["world"]
            w = entry["w"]
            rigid = entry["rigid"]

            for vi in range(entry["n"]):
                if rigid[vi]:
                    continue

                pt = Vector(world[vi])
                hits = kd.find_range(pt, tol)
                if not hits:
                    continue

                max_w = float(w[vi])
                for _, hit_idx, _ in hits:
                    hw = active_items[hit_idx][3]
                    if hw > max_w:
                        max_w = hw

                if abs(w[vi] - max_w) > 1e-4:
                    updates_per_obj[o_idx][vi] = max_w

        for o_idx, updates in updates_per_obj.items():
            if not updates:
                continue
            entry = obj_entries[o_idx]
            obj = entry["obj"]
            vg = entry["vg"] or obj.vertex_groups.new(name=vgroup_name)

            for vi, new_w in updates.items():
                if new_w > 0.0005:
                    vg.add([vi], max(0.0, min(1.0, new_w)), 'REPLACE')
                elif new_w <= 0.0001:
                    vg.remove([vi])
                synced_count += 1

    return synced_count


def smooth_boundary_gradient_taper(target_objs, vg_names, max_delta=0.08, passes=3):
    """
    물리 본 경계면에서 급격한 가중치 절벽(예: 0.35 -> 0.00)으로 인해 폴리곤이 찢어지는
    현상을 방지하기 위해, 연결된 메쉬 엣지(Edge)를 따라 매끄러운 그라데이션 감쇠(Taper)를 적용합니다.
    어떤 단일 엣지도 max_delta를 초과하는 가중치 급변을 겪지 않도록 보장합니다.
    """
    valid_objs = [o for o in target_objs if o and o.type == 'MESH' and len(o.data.vertices) > 0]
    if not valid_objs or not vg_names:
        return 0

    total_smoothed = 0
    for obj in valid_objs:
        mesh = obj.data
        n_verts = len(mesh.vertices)
        if n_verts == 0 or len(mesh.edges) == 0:
            continue

        vg_rigid = obj.vertex_groups.get(VG_RIGID)
        r_arr = np.zeros(n_verts, dtype=bool)
        if vg_rigid:
            r_data = read_vertex_group_arrays(obj, {vg_rigid.index: "r"}).get("r")
            if r_data is not None:
                r_arr = (r_data > 0.5)

        # 엣지 인접 리스트 구축
        adj = [[] for _ in range(n_verts)]
        for e in mesh.edges:
            v1, v2 = e.vertices[0], e.vertices[1]
            adj[v1].append(v2)
            adj[v2].append(v1)

        for vg_name in vg_names:
            vg = obj.vertex_groups.get(vg_name)
            if not vg:
                continue

            arrs = read_vertex_group_arrays(obj, {vg.index: "w"})
            w = arrs.get("w", np.zeros(n_verts, dtype=np.float32)).copy()
            if w.max() <= 0.005:
                continue

            changed = False
            for _ in range(passes):
                active_idx = np.nonzero((w > 0.01) & ~r_arr)[0]
                for vi in active_idx:
                    cur_w = float(w[vi])
                    neighbors = adj[vi]
                    if not neighbors:
                        continue
                    min_nb = min(float(w[nb]) for nb in neighbors)
                    diff = cur_w - min_nb
                    if diff > max_delta:
                        # 경계면 완만화: 절벽 정점의 가중치를 부드럽게 감쇠
                        w[vi] = cur_w - (diff - max_delta) * 0.45
                        changed = True
                        total_smoothed += 1

            if changed:
                for vi in range(n_verts):
                    if r_arr[vi]:
                        continue
                    new_val = float(w[vi])
                    if new_val > 0.0005:
                        vg.add([vi], max(0.0, min(1.0, new_val)), 'REPLACE')
                    else:
                        vg.remove([vi])

    return total_smoothed


def enforce_radial_monotonic_non_penetration(target_objs, bone_info, active_bones, search_radius=0.015):
    """
    [절대 다층 방사형 비침투 제약 (Strict Radial Multi-Layer Non-Penetration Constraint)]:
    "본 중심에 가까운 안쪽 메쉬(속살)는 본 중심에서 더 먼 바깥쪽 메쉬(겉옷)를 절대로 넘지 못한다 (w_inner <= w_outer)."
    
    기하학적 원리:
    1. 본의 흉벽 앵커(Head)를 원점으로 하여, 각 정점의 방사 거리 r = ||P - Head|| 및 방사 방향 단위벡터 u = (P - Head)/r 산출.
    2. 본 중심에서 바라본 동일 방사선(Ray, 각도 10도 이내 또는 각거리 3.5mm 이내) 선상에 놓인 정점들을 매칭.
    3. 방사 거리 차이가 0.8mm ~ 15mm인 서로 다른 층(속살 vs 겉옷)에 대해:
       - r_inner < r_outer (본 중심에 가까운 살 vs 먼 옷)
       - 절대 비침투: w_inner <= w_outer
       - 만약 w_inner > w_outer 이면:
         - 겉옷(outer)은 최소한 속살 이상으로 상향 일치: w_outer = max(w_outer, w_inner)
         - 속살(inner)은 겉옷을 넘지 못하도록 엄격 제한: w_inner = min(w_inner, w_outer)
    4. 표면 법선(Normal) 오차에 취약하지 않고 곡면 전체에서 속살 돌출을 100% 원천 봉쇄합니다.
    """
    if not target_objs or not active_bones or not bone_info:
        return 0

    valid_objs = [o for o in target_objs if o and o.type == 'MESH' and len(o.data.vertices) > 0]
    if not valid_objs:
        return 0

    total_adjusted = 0

    for bname in active_bones:
        bentry = bone_info.get(bname)
        if not bentry:
            b_low = bname.lower()
            for k, e in bone_info.items():
                if k.lower() == b_low or e["bname"].lower() == b_low:
                    bentry = e
                    break
        if not bentry:
            continue

        head = Vector(bentry["head"])
        vg_name = f"XXMI_{bname}"

        # 본의 대상 정점, 월드 좌표 및 참조 수집
        all_pts = []
        all_weights = []
        all_refs = []  # (obj, vi)

        for obj in valid_objs:
            vg = obj.vertex_groups.get(vg_name)
            if not vg:
                continue
            world = world_vertex_positions(obj)
            arrs = read_vertex_group_arrays(obj, {vg.index: "w"})
            w_arr = arrs.get("w", np.zeros(len(world), dtype=np.float32))

            e0, e1 = mesh_edge_arrays(obj.data)
            comp_labels = label_components(len(world), e0, e1)

            for vi in range(len(world)):
                w = float(w_arr[vi])
                if w > 0.0005:
                    all_pts.append(Vector(world[vi]))
                    all_weights.append(w)
                    all_refs.append((obj, comp_labels[vi], vi))

        if len(all_pts) < 2:
            continue

        weights_arr = np.array(all_weights, dtype=np.float32)
        n_pts = len(all_pts)

        # 본 앵커로부터의 방사 벡터 및 거리 계산
        dists = np.empty(n_pts, dtype=np.float64)
        uvecs = []
        for k in range(n_pts):
            v_head = all_pts[k] - head
            d = v_head.length
            dists[k] = d
            uvecs.append(v_head / max(d, 1e-6))

        kd = KDTree(n_pts)
        for k, p in enumerate(all_pts):
            kd.insert(p, k)
        kd.balance()

        for pass_idx in range(2):
            orig_weights = weights_arr.copy()
            promotions = {}
            clamps = {}

            for k in range(n_pts):
                cur_w = float(orig_weights[k])
                if cur_w <= 0.005:
                    continue
                cur_p = all_pts[k]
                cur_d = dists[k]
                cur_u = uvecs[k]

                hits = kd.find_range(cur_p, search_radius)
                obj_k, comp_k, vi_k = all_refs[k]

                for _, hit_k, _ in hits:
                    if hit_k == k:
                        continue
                    obj_hit, comp_hit, vi_hit = all_refs[hit_k]

                    # 동일 오브젝트의 동일 연결 컴포넌트(연속된 메쉬 표면) 내부의 정점끼리는 비교하지 않음 (표면 감쇠 보존)
                    if obj_k == obj_hit and comp_k == comp_hit:
                        continue

                    hit_d = dists[hit_k]
                    diff_d = hit_d - cur_d
                    abs_diff_d = abs(diff_d)

                    # 1. 서로 다른 층(Shell) 사이인지 확인 (0.8mm ~ search_radius)
                    if abs_diff_d < 0.0008 or abs_diff_d > search_radius:
                        continue

                    # 2. 동일한 방사 각도 선상에 놓여 있는지 확인 (내적 >= 0.985 또는 접평면 오차 <= 3.5mm)
                    hit_u = uvecs[hit_k]
                    dot_u = cur_u.dot(hit_u)
                    if dot_u < 0.985:
                        continue

                    hit_w = float(orig_weights[hit_k])

                    if diff_d > 0:
                        # hit_k가 cur_p보다 바깥쪽(겉옷): cur_p(살) <= hit_k(옷)
                        # 겉옷(outer)은 최소한 속살(inner) 이상으로 움직이도록 상향 일치
                        if cur_w > hit_w:
                            promotions[hit_k] = max(promotions.get(hit_k, hit_w), cur_w)
                    else:
                        # hit_k가 cur_p보다 안쪽(속살): hit_k(살) <= cur_p(옷)
                        # 겉옷(cur_p)은 최소한 속살(hit_k) 이상으로 움직이도록 상향 일치
                        if hit_w > cur_w:
                            promotions[k] = max(promotions.get(k, cur_w), hit_w)

            # 비연쇄적 일괄 반영
            changed = False
            for vi_k, val in promotions.items():
                if val > weights_arr[vi_k]:
                    weights_arr[vi_k] = val
                    total_adjusted += 1
                    changed = True
            for vi_k, val in clamps.items():
                if val < weights_arr[vi_k]:
                    weights_arr[vi_k] = val
                    total_adjusted += 1
                    changed = True

            if not changed:
                break

        # 버텍스 그룹에 반영
        updates_per_obj = {}
        for k in range(n_pts):
            obj, comp_id, vi = all_refs[k]
            new_w = float(weights_arr[k])
            if abs(new_w - all_weights[k]) > 1e-4:
                updates_per_obj.setdefault(obj, []).append((vi, new_w))

        for obj, v_list in updates_per_obj.items():
            vg = obj.vertex_groups.get(vg_name)
            if not vg:
                continue
            for vi, new_w in v_list:
                if new_w > 0.0005:
                    vg.add([vi], max(0.0, min(1.0, new_w)), 'REPLACE')
                else:
                    vg.remove([vi])

    return total_adjusted


def rebake_mesh_weights_from_rig(context, rig, props=None):
    """
    물리 리그(XXMI_Physics_Rig) 내의 본 위치를 기준으로
    씬 내의 모든 관련 메쉬 오브젝트들의 버텍스 가중치를 동적으로 다시 계산합니다.
    사용자가 등록한 개별 모핑 본 슬롯 및 본 위치 수정을 100% 반영합니다.
    """
    if props is None:
        props = context.scene.xxmi_custom_physics_props

    ensure_all_objects_in_object_mode(context)

    bone_info = get_rig_bone_info(rig)
    if not bone_info:
        return 0, "유효한 물리 본을 찾을 수 없습니다."

    # 1. 활성 본 슬롯 목록 추출
    active_bones = []
    if props and hasattr(props, "bone_items") and len(props.bone_items) > 0:
        active_bones = [it.name for it in props.bone_items if it.name.strip()]

    if not active_bones:
        active_bones = [k for k in bone_info.keys() if k not in ("L", "R")]

    if not active_bones:
        active_bones = list(bone_info.keys())

    # 2. 대상 메쉬 오브젝트 수집
    active_vg_names = [f"XXMI_{bn}" for bn in active_bones]
    legacy_vg_names = [VG_BREAST_L, VG_BREAST_R]
    all_target_vg_names = set(active_vg_names + legacy_vg_names + [VG_RIGID])

    target_objs = [
        o for o in bpy.data.objects
        if o.type == 'MESH' and any(o.vertex_groups.get(vgn) for vgn in all_target_vg_names)
    ]

    if not target_objs:
        target_objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not target_objs and context.active_object and context.active_object.type == 'MESH':
            target_objs = [context.active_object]
        if not target_objs:
            target_objs = [o for o in bpy.data.objects if o.type == 'MESH']

    if not target_objs:
        return 0, "씬 내에서 대상 메쉬 오브젝트를 찾을 수 없습니다."

    falloff_p = getattr(props, "falloff_power", 1.0)
    peak_w = getattr(props, "peak_weight", 0.40)
    smooth_iters = getattr(props, "smooth_iterations", 2)
    lock_registered = getattr(props, "lock_registered_verts", True)
    updated_obj_count = 0
    total_recomputed_verts = 0

    obj_data = {}
    for obj in target_objs:
        ensure_object_visible(context, obj)

        vg_rigid = obj.vertex_groups.get(VG_RIGID)
        world = world_vertex_positions(obj)
        n_verts = len(world)

        rigid_mask = np.zeros(n_verts, dtype=bool)
        if vg_rigid:
            r_arr = read_vertex_group_arrays(obj, {vg_rigid.index: "rigid"}).get("rigid")
            if r_arr is not None:
                rigid_mask = (r_arr > 0.5)

        if vg_rigid and np.any(rigid_mask):
            e0, e1 = mesh_edge_arrays(obj.data)
            lab_all = label_components(n_verts, e0, e1)
            uniq_labs, lab_counts = np.unique(lab_all, return_counts=True)
            to_add_list = []
            for lab_id, cnt in zip(uniq_labs, lab_counts):
                if cnt <= 2500:
                    isl_mask = (lab_all == lab_id)
                    n_rig = np.count_nonzero(rigid_mask & isl_mask)
                    if n_rig >= 5 or (n_rig / cnt) >= 0.15:
                        diff = np.nonzero(isl_mask & ~rigid_mask)[0]
                        if len(diff) > 0:
                            rigid_mask[isl_mask] = True
                            to_add_list.extend(diff.tolist())
            if to_add_list:
                vg_rigid.add(to_add_list, 1.0, 'REPLACE')

        obj_data[obj] = {
            "world": world,
            "rigid_mask": rigid_mask,
        }

    updated_obj_count = len(obj_data)

    # 3. 각 모핑 본별 독립 가중치 계산 (슬롯별 설정 및 본 위치 반영)
    for bname in active_bones:
        bentry = bone_info.get(bname)
        if not bentry:
            b_low = bname.lower()
            for k, e in bone_info.items():
                if k.lower() == b_low or e["bname"].lower() == b_low:
                    bentry = e
                    break
        if not bentry:
            continue

        b_params = get_bone_params(props, bname)
        side_falloff_p = b_params["falloff_power"]
        side_peak_w = b_params["peak_weight"]
        side_lock_registered = b_params["lock_registered_verts"]

        cent = np.array(bentry["centroid"], dtype=np.float64)
        head = np.array(bentry["head"], dtype=np.float64)
        tail = np.array(bentry["tail"], dtype=np.float64)
        fwd_u = np.array(bentry["fwd_unit"], dtype=np.float64)
        radius = bentry["radius"]
        eff_radius = max(0.02, radius)

        vg_target_name = f"XXMI_{bname}"

        # 해당 슬롯의 정점 사전 데이터 가져오기
        b_item = get_bone_item(props, bname)
        b_item_verts = get_bone_data(b_item) if b_item else {}

        side_cand = []
        zero_entries = []

        for obj, odata in obj_data.items():
            vg = obj.vertex_groups.get(vg_target_name) or obj.vertex_groups.new(name=vg_target_name)
            world = odata["world"]
            rigid_mask = odata["rigid_mask"]
            n_v = len(world)

            slot_obj_verts = set(b_item_verts.get(obj.name, []))

            cur_vg = obj.vertex_groups.get(vg_target_name)
            cur_arr = read_vertex_group_arrays(obj, {cur_vg.index: "cur"})["cur"] if cur_vg else np.zeros(n_v)
            is_registered_b = (cur_arr > 0.0001)
            for vi in slot_obj_verts:
                if vi < n_v:
                    is_registered_b[vi] = True

            if slot_obj_verts:
                # 사용자가 등록한 정점 목록이 있을 경우 해당 정점들을 엄격히 존중 (주변 메쉬로의 무단 번짐 차단)
                cand_mask = np.zeros(n_v, dtype=bool)
                for vi in slot_obj_verts:
                    if vi < n_v:
                        cand_mask[vi] = True
                cand_mask &= ~rigid_mask
            else:
                # 사용자가 별도 등록하지 않은 연관 오브젝트(겉옷/장신구 등)의 경우 본 기하학 반경 내 정점 자동 후보군 포함
                dist = np.linalg.norm(world - cent, axis=1)
                in_radius = (dist <= eff_radius)
                dot_fwd = (world - head) @ fwd_u
                front_ok = (dot_fwd >= -eff_radius * 0.25)
                cand_mask = (in_radius & front_ok) & ~rigid_mask

            cand_idx = np.nonzero(cand_mask)[0]
            for vi in cand_idx.tolist():
                side_cand.append((obj, vg, vi, world[vi]))

        if not side_cand:
            continue

        cand_pts = np.array([c[3] for c in side_cand], dtype=np.float64)
        baked_w = calculate_anatomical_weights_with_anti_clipping(
            cand_pts, head, tail, eff_radius, side_falloff_p, side_peak_w
        )

        for (obj, vg, vi, _), wi in zip(side_cand, baked_w.tolist()):
            if wi > 0.0005:
                vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')
                total_recomputed_verts += 1
            else:
                vg.remove([vi])

        for vg, z_list in zero_entries:
            vg.remove(z_list)

    # 4. 라플라시안 스무딩 (모든 활성 모핑 본 버텍스 그룹 대상)
    for bname in active_bones:
        b_params = get_bone_params(props, bname)
        s_iters = b_params["smooth_iterations"]
        if s_iters > 0:
            b_item = get_bone_item(props, bname)
            b_item_verts = get_bone_data(b_item) if b_item else {}
            vg_name = f"XXMI_{bname}"
            for obj, odata in obj_data.items():
                vg = obj.vertex_groups.get(vg_name)
                if vg:
                    slot_obj_verts = set(b_item_verts.get(obj.name, []))
                    n_v = len(odata["world"])
                    allowed_m = np.zeros(n_v, dtype=bool)
                    for vi in slot_obj_verts:
                        if vi < n_v:
                            allowed_m[vi] = True
                    cur_arr = read_vertex_group_arrays(obj, {vg.index: "w"}).get("w", np.zeros(n_v))
                    allowed_m |= (cur_arr > 0.0001)

                    smooth_mesh_weights(
                        obj, [vg], odata["rigid_mask"],
                        iters=s_iters, alpha=0.45, lock_registered=lock_registered,
                        allowed_mask=allowed_m
                    )

    target_peak_cap = float(peak_w)

    # 5. 다층 레이어 절대 비침투 제약 (Strict Multi-Layer Non-Penetration Constraint):
    # 본 중심에 가까운 안쪽 메쉬(속살)는 바깥쪽 메쉬(겉옷)를 넘지 못하도록 보장 (w_outer >= w_inner, w_inner <= w_outer)
    enforce_radial_monotonic_non_penetration(
        target_objs, bone_info, active_bones, search_radius=0.015
    )

    # 7. 강체(액세서리) 표면 밀착 가중치 동기화
    slot_exclude = {}
    for item in props.rigid_items:
        rdata = get_rigid_data(item)
        if not rdata:
            continue
        for oname, vis in rdata.items():
            slot_exclude.setdefault(oname, set()).update(vis)

    # 레거시 / 슬롯 외 강체 정점 fallback 클러스터 통합
    unify_rigid_clusters_across_objects(
        target_objs, bone_info, falloff_p, peak_w, exclude_verts_by_obj=slot_exclude
    )

    # 활성 본별 주변 연질(Soft) 표면 KD-Tree 구성
    bone_kd_map = {}
    for bname in active_bones:
        vg_name = f"XXMI_{bname}"
        soft_pts = []
        soft_weights = []
        for obj, odata in obj_data.items():
            world = odata["world"]
            rigid_mask = odata["rigid_mask"]
            vg = obj.vertex_groups.get(vg_name)
            if not vg:
                continue
            cur_arrs = read_vertex_group_arrays(obj, {vg.index: "w"})
            w_arr = cur_arrs["w"]
            for vi in range(len(world)):
                if not rigid_mask[vi] and w_arr[vi] > 0.001:
                    soft_pts.append(Vector(world[vi]))
                    soft_weights.append(float(w_arr[vi]))

        if soft_pts:
            kd = KDTree(len(soft_pts))
            for k, p in enumerate(soft_pts):
                kd.insert(p, k)
            kd.balance()
            bone_kd_map[bname] = (kd, soft_weights)

    # 다중 강체 슬롯 가중치 계산 및 할당 (통합 표면 간접 추종 / Universal Surface Proximity Transfer)
    # 단일 본 부품(단추/배지) 및 다중 본 브릿지 부품(앞쪽 끈/리본 매듭), 다중 오브젝트 결합 부품(무전기+케이스) 모두 지원
    applied_slots = 0
    for item in props.rigid_items:
        rdata = get_rigid_data(item)
        if not rdata:
            continue

        w_scale = getattr(item, "weight_scale", 1.0)

        # 1. 해당 강체 슬롯에 속한 모든 오브젝트의 정점 및 3D 월드 좌표 수집
        slot_objs_data = []
        all_pts_slot = []
        for oname, vis in rdata.items():
            robj = bpy.data.objects.get(oname)
            if not robj or robj.type != 'MESH' or not vis:
                continue
            r_world = world_vertex_positions(robj)
            slot_objs_data.append((robj, vis, r_world))
            for vi in vis:
                if vi < len(r_world):
                    all_pts_slot.append(Vector(r_world[vi]))

        if not all_pts_slot:
            continue

        # 2. 강체 슬롯 전체(다중 오브젝트 포함)를 하나의 완전한 일체형 강체로 취급하여
        #    하부 소프트 표면과의 밀착 접촉면(Mounting Base)을 기준으로 단일 통합 가중치 벡터 산출
        slot_unified_weights = {bn: 0.0 for bn in active_bones}

        for bn in active_bones:
            if bn not in bone_kd_map:
                continue
            kd, s_weights = bone_kd_map[bn]

            dists_and_weights = []
            for pt in all_pts_slot:
                res = kd.find(pt)
                if res is not None and res[0] is not None:
                    dists_and_weights.append((res[2], s_weights[res[1]]))

            if dists_and_weights:
                min_d = min(d for d, _ in dists_and_weights)
                if min_d < 0.08:
                    # 강체 부품이 신체/의상에 맞닿아 있는 접촉면(최단 거리 근방 15mm 이내) 정점들의 가중치를 기준으로 슬롯 가중치 결정
                    contact_pts = [(d, w) for d, w in dists_and_weights if d <= min_d + 0.015]
                    inv_d_sum = sum(1.0 / max(d, 0.002) for d, _ in contact_pts)
                    base_w = sum(w * (1.0 / max(d, 0.002)) for d, w in contact_pts) / max(1e-6, inv_d_sum)
                    dist_factor = max(0.0, 1.0 - max(0.0, min_d - 0.04) / 0.04)
                    slot_unified_weights[bn] = base_w * dist_factor * w_scale

        # 3. 강체 슬롯에 속한 모든 오브젝트(예: Hair의 무전기 본체, Body의 무전기 케이스)의
        #    모든 정점에 100.000% 완벽히 동일한 단일 가중치 부여 (상대 변위 0 -> 뚫림 원천 차단)
        for robj, vis, _ in slot_objs_data:
            vg_rigid = robj.vertex_groups.get(VG_RIGID) or robj.vertex_groups.new(name=VG_RIGID)
            vg_rigid.add(vis, 1.0, 'REPLACE')

            for bn in active_bones:
                vg_target = robj.vertex_groups.get(f"XXMI_{bn}") or robj.vertex_groups.new(name=f"XXMI_{bn}")
                slot_w = float(slot_unified_weights.get(bn, 0.0))
                if slot_w > 0.001:
                    vg_target.add(vis, min(1.0, slot_w), 'REPLACE')
                else:
                    vg_target.remove(vis)

        applied_slots += 1

    # 7.5. 경계면 완만화(Boundary Taper)로 절벽 찢어짐 원천 차단
    active_vg_names = [f"XXMI_{bn}" for bn in active_bones]
    smooth_boundary_gradient_taper(target_objs, active_vg_names, max_delta=0.08, passes=3)

    # 7.6. 가슴 중앙선 중복 가중치 재정규화 (가슴골 2배 돌출 원천 차단)
    normalize_overlapping_bone_weights(
        target_objs, active_bones, max_total_weight=target_peak_cap
    )

    # 7.7. [절대 최종 단계]: 모든 동일 위치 공유 정점(UV 심/하드 엣지) 100% 완전 동기화
    num_synced = synchronize_coincident_vertices_across_objects(
        target_objs, vg_names=active_vg_names, tol=0.0002
    )

    return updated_obj_count, f"{updated_obj_count}개 메쉬 ({total_recomputed_verts}개 정점, 동기화 {num_synced}개, 강체슬롯 {applied_slots}개) 가중치 갱신 완료"


