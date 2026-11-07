import math
import numpy as np
from collections import deque
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree
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


def dilate_mask_topological(mask, e0, e1, steps=2):
    """메쉬 엣지 연결성을 따라 불리언 마스크를 지정된 링(step) 수만큼 확장합니다."""
    out = mask.copy()
    if len(e0) == 0 or steps <= 0:
        return out
    for _ in range(steps):
        nxt = out.copy()
        nxt[e0] |= out[e1]
        nxt[e1] |= out[e0]
        out = nxt
    return out


def calculate_bridged_pair_weights(
    world_pts,
    head_a, tail_a, radius_a, falloff_a, peak_a,
    head_b, tail_b, radius_b, falloff_b, peak_b,
):
    """
    인접한 두 모핑 본(예: 좌/우 가슴) 사이의 중첩/브릿지 영역에서
    3D Lofted Capsule(리본 엔벨로프)과 Raised-Cosine Partition of Unity를 결합하여
    가슴 중앙이 꺼지거나 부자연스럽게 꺾이지 않고 일체감 있게 연동되도록 (w_a, w_b)를 계산합니다.
    """
    if len(world_pts) == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    w_a_indiv = calculate_anatomical_weights_with_anti_clipping(
        world_pts, head_a, tail_a, radius_a, falloff_a, peak_a
    )
    w_b_indiv = calculate_anatomical_weights_with_anti_clipping(
        world_pts, head_b, tail_b, radius_b, falloff_b, peak_b
    )

    ha = np.array(head_a, dtype=np.float64)
    ta = np.array(tail_a, dtype=np.float64)
    hb = np.array(head_b, dtype=np.float64)
    tb = np.array(tail_b, dtype=np.float64)

    u_ab = hb - ha
    d_ab = float(np.linalg.norm(u_ab))
    if d_ab < 1e-4:
        return w_a_indiv, w_b_indiv

    u_dir = u_ab / d_ab
    ra = max(0.02, float(radius_a))
    rb = max(0.02, float(radius_b))

    # 1. 두 본을 잇는 축 상의 정규화 위치 s (s=0: 본 A 중심, s=0.5: 중앙선, s=1: 본 B 중심)
    s = ((world_pts - ha) @ u_dir) / d_ab
    s_c = np.clip(s, 0.0, 1.0)

    # 2. s_c 위치에서의 보간된 브릿지 본 세그먼트 [H(s_c) -> T(s_c)]
    h_s = (1.0 - s_c)[:, None] * ha[None, :] + s_c[:, None] * hb[None, :]
    t_s = (1.0 - s_c)[:, None] * ta[None, :] + s_c[:, None] * tb[None, :]
    r_s = (1.0 - s_c) * ra + s_c * rb
    peak_s = (1.0 - s_c) * float(peak_a) + s_c * float(peak_b)
    falloff_s = (1.0 - s_c) * float(falloff_a) + s_c * float(falloff_b)

    vec_s = t_s - h_s
    len_s = np.linalg.norm(vec_s, axis=1)
    safe_len_s = np.maximum(1e-5, len_s)
    unit_s = vec_s / safe_len_s[:, None]

    # 전방 깊이 감쇠 (맨살 가슴골처럼 흉벽에 붙어 있으면 t_proj ~ 0 이 되어 자동으로 0 유지,
    # 일체형 셔츠처럼 앞으로 떠 있으면 t_proj > 0 이 되어 자연스러운 볼륨 유지)
    pts_rel = world_pts - h_s
    t_proj = np.sum(pts_rel * unit_s, axis=1)
    t_clamped = np.clip(t_proj / safe_len_s, 0.0, 1.0)
    w_depth = np.where(t_proj > 0.0, 0.5 * (1.0 - np.cos(np.pi * t_clamped)), 0.0)

    # 브릿지 리본 표면으로부터의 3D 수직 거리 및 방사형 Hann 코사인 감쇠
    proj_pts = h_s + t_clamped[:, None] * vec_s
    d_perp = np.linalg.norm(world_pts - proj_pts, axis=1)
    max_r_s = r_s * 1.30
    rho_norm = np.clip(d_perp / np.maximum(1e-5, max_r_s), 0.0, 1.0)
    w_radial = np.where(d_perp < max_r_s, 0.5 * (1.0 + np.cos(np.pi * rho_norm)), 0.0)

    eff_power = 0.35 * np.maximum(0.05, falloff_s)
    w_loft = peak_s * np.power(np.maximum(0.0, w_depth * w_radial), eff_power)
    w_loft = np.where((w_radial > 0.0) & (w_depth > 0.0), np.clip(w_loft, 0.0, 1.0), 0.0)

    # 3. 본 반경(radius_a + radius_b)에 비례하는 브릿지 장력 보존 및 크로스 블렌딩 폭 결정
    reach_ratio = (1.30 * (ra + rb)) / d_ab
    bridge_strength = float(np.clip((reach_ratio - 0.85) / 0.55, 0.0, 1.0))

    w_indiv_sum = np.minimum(peak_s, w_a_indiv + w_b_indiv)
    w_total = (1.0 - bridge_strength) * w_indiv_sum + bridge_strength * np.maximum(w_indiv_sum, w_loft)

    # 중앙(s=0.5) 기준 좌/우 Raised-Cosine Partition of Unity (alpha_a + alpha_b == 1.0)
    sigma = float(np.clip(0.42 * ((ra + rb) / d_ab), 0.12, 0.90))
    tau = np.clip((s - (0.5 - sigma)) / max(1e-4, 2.0 * sigma), 0.0, 1.0)
    alpha_b = 0.5 * (1.0 - np.cos(np.pi * tau))
    alpha_a = 1.0 - alpha_b

    w_a_out = np.where(w_total > 0.0005, alpha_a * w_total, 0.0)
    w_b_out = np.where(w_total > 0.0005, alpha_b * w_total, 0.0)

    # 동일 3D 좌표(UV 심/하드 엣지) 0.5mm 이내 정점 동기화
    if len(world_pts) > 1:
        kd = KDTree(len(world_pts))
        for k, pt in enumerate(world_pts.tolist()):
            kd.insert(Vector(pt), k)
        kd.balance()
        for idx in range(len(world_pts)):
            co = Vector(world_pts[idx])
            hits = kd.find_range(co, 0.0005)
            if len(hits) > 1:
                max_wa = max(w_a_out[h_idx] for _, h_idx, _ in hits)
                max_wb = max(w_b_out[h_idx] for _, h_idx, _ in hits)
                for _, h_idx, _ in hits:
                    w_a_out[h_idx] = max_wa
                    w_b_out[h_idx] = max_wb

    return w_a_out, w_b_out


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
    강체(VG_RIGID) 정점과 맞닿은 연질(Soft) 정점은 강체 가중치를 우선 추종하되
    연질 정점의 VG_RIGID 플래그는 건드리지 않습니다.
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

            mask = (w > 0.0001) | rigid
            for vi in np.nonzero(mask)[0].tolist():
                active_items.append((o_idx, vi, world[vi], float(w[vi]), bool(rigid[vi])))

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

                rigid_hits = [active_items[h_idx][3] for _, h_idx, _ in hits if active_items[h_idx][4]]
                if rigid_hits:
                    target_w = float(rigid_hits[0])
                else:
                    target_w = float(w[vi])
                    for _, hit_idx, _ in hits:
                        hw = active_items[hit_idx][3]
                        if hw > target_w:
                            target_w = hw

                if abs(w[vi] - target_w) > 1e-4:
                    updates_per_obj[o_idx][vi] = target_w

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


def smooth_boundary_gradient_taper(target_objs, vg_names, max_delta=0.08, passes=3, seam_locked_by_obj=None):
    """
    물리 본 경계면에서 급격한 가중치 절벽(예: 0.35 -> 0.00)이나 강체-연질 봉합선(Seam)의
    급격한 단차로 인해 폴리곤이 찢어지는 현상을 방지하기 위해,
    연결된 메쉬 엣지(Edge)를 따라 양방향 그라데이션 완만화(Taper)를 적용합니다.
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

        seam_locked = None
        if seam_locked_by_obj and obj in seam_locked_by_obj:
            seam_locked = seam_locked_by_obj[obj]
        if seam_locked is None:
            seam_locked = np.zeros(n_verts, dtype=bool)

        locked_mask = r_arr | seam_locked

        # 강체-연질 봉합선(seam_locked) 주변 6엣지 이내 영역만 상향 완만화(Upward Taper) 허용
        # (일반 외곽 경계의 w=0 체간부 정점까지 상향 견인하면 외곽 경계가 들뜨고 곡면 굴절이 발생하므로 차단)
        if np.any(seam_locked):
            e0, e1 = mesh_edge_arrays(mesh)
            near_seam_mask = dilate_mask_topological(seam_locked, e0, e1, steps=6) & ~locked_mask
        else:
            near_seam_mask = np.zeros(n_verts, dtype=bool)

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
                # 1. 상향 완만화: 강체-연질 봉합선(seam_locked) 인접 연질 정점이 봉합선으로 부드럽게 이어지도록 보간
                if np.any(near_seam_mask):
                    w_up = w.copy()
                    for vi in np.nonzero(near_seam_mask)[0]:
                        neighbors = adj[vi]
                        if not neighbors:
                            continue
                        max_nb = max(float(w_up[nb]) for nb in neighbors)
                        cur_w = float(w_up[vi])
                        if max_nb - cur_w > max_delta:
                            w[vi] = cur_w + (max_nb - cur_w - max_delta) * 0.45
                            changed = True
                            total_smoothed += 1

                # 2. 하향 완만화: 외곽 경계면에서 절벽 정점의 가중치를 부드럽게 감쇠 (봉합선 근방은 제외)
                w_prev = w.copy()
                active_idx = np.nonzero((w_prev > 0.005) & ~locked_mask & ~near_seam_mask)[0]
                for vi in active_idx:
                    cur_w = float(w_prev[vi])
                    neighbors = adj[vi]
                    if not neighbors:
                        continue
                    min_nb = min(float(w_prev[nb]) for nb in neighbors)
                    diff = cur_w - min_nb
                    if diff > max_delta:
                        w[vi] = cur_w - (diff - max_delta) * 0.45
                        changed = True
                        total_smoothed += 1

            if changed:
                for vi in range(n_verts):
                    if locked_mask[vi]:
                        continue
                    new_val = float(w[vi])
                    if new_val > 0.0005:
                        vg.add([vi], max(0.0, min(1.0, new_val)), 'REPLACE')
                    else:
                        vg.remove([vi])

    return total_smoothed


def _barycentric_weights_3d(p, a, b, c):
    """3D 공간의 점 p를 삼각형 (a, b, c) 평면에 투영한 무게중심 좌표 (u, v, w)를 반환합니다."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    w = max(0.0, min(1.0, w))
    s = u + v + w
    if s > 1e-8:
        return u / s, v / s, w / s
    return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0


def enforce_radial_monotonic_non_penetration(target_objs, bone_info, active_bones, search_radius=0.014):
    """
    [BVH 삼각망 무게중심 투영 기반 다층 쉘 등각 동기화 (Conformal Multi-Shell Synchronization)]:
    3D 스프링은 전방(-Z), 후방(+Z), 상하(±Y), 좌우(±X) 모든 방향으로 진동하므로,
    밀착된 다층 메쉬(0.3mm ~ 8mm 간격의 속살과 겉옷) 사이에서 가중치 벡터가 조금이라도 다르면
    전진 시 속살이 뚫고 나오거나 후진/하강 시 겉옷이 속살 안으로 파고드는 관통이 발생합니다.

    해결 원리:
    1. 각 연결 컴포넌트(Mesh Island)별로 BVHTree 삼각망을 구축합니다 (경계면 w=0 폴리곤 포함).
    2. 본 앵커(Head)에서 바깥쪽을 향하는 외향 법선(Outward Normal) 기준으로 컴포넌트 간
       피복 관계(Outer Shell vs Inner Shell)를 비대칭 판별합니다.
    3. 가장 바깥쪽 쉘(겉옷)에서 안쪽 쉘(속살) 순서로, 안쪽 정점 P_inner에서 가장 가까운
       바깥쪽 삼각형 위 투영점의 무게중심 가중치(Barycentric Weight Vector)를 전 본에 대해 동시 전사합니다.
    4. 의상 외곽 경계선 밖으로 노출되는 피부는 접평면 거리(d_tang: 2.5mm -> 8.0mm)에 따라
       부드럽게 원래 피부 가중치로 블렌딩하여 경계 단차와 관통을 동시에 0으로 만듭니다.
    """
    if not target_objs or not active_bones or not bone_info:
        return 0

    valid_objs = [o for o in target_objs if o and o.type == 'MESH' and len(o.data.vertices) > 0]
    if not valid_objs:
        return 0

    bone_heads = []
    bone_radii = []
    for bname in active_bones:
        bentry = bone_info.get(bname)
        if not bentry:
            b_low = bname.lower()
            for k, e in bone_info.items():
                if k.lower() == b_low or e["bname"].lower() == b_low:
                    bentry = e
                    break
        if bentry:
            bone_heads.append(Vector(bentry["head"]))
            bone_radii.append(max(0.02, float(bentry.get("radius", 0.08))))

    if not bone_heads:
        return 0

    num_bones = len(active_bones)
    vg_names = [f"XXMI_{bn}" for bn in active_bones]

    # 오브젝트별 월드 좌표, 외향 법선, 본 가중치 행렬, 컴포넌트, 삼각망 수집
    obj_states = []
    cand_shells = {}  # key: (o_idx, comp_id)

    for o_idx, obj in enumerate(valid_objs):
        mesh = obj.data
        mesh.calc_loop_triangles()
        world = world_vertex_positions(obj)
        n_v = len(world)
        if n_v == 0:
            continue

        out_dirs = []
        near_bone_dist = np.full(n_v, 1e9, dtype=np.float64)
        for vi in range(n_v):
            p_v = Vector(world[vi])
            best_h = bone_heads[0]
            best_d = 1e9
            for bh in bone_heads:
                d = (p_v - bh).length
                if d < best_d:
                    best_d = d
                    best_h = bh
            near_bone_dist[vi] = best_d
            od = (p_v - best_h)
            out_dirs.append(od.normalized() if od.length > 1e-8 else Vector((0.0, -1.0, 0.0)))

        gmap = {}
        vg_rigid = obj.vertex_groups.get(VG_RIGID)
        if vg_rigid:
            gmap[vg_rigid.index] = "rigid"
        for bi, vgn in enumerate(vg_names):
            vg = obj.vertex_groups.get(vgn)
            if vg:
                gmap[vg.index] = f"b_{bi}"

        arrs = read_vertex_group_arrays(obj, gmap)
        rigid_mask = (arrs.get("rigid", np.zeros(n_v, dtype=np.float32)) > 0.5)
        w_mat = np.zeros((num_bones, n_v), dtype=np.float32)
        for bi in range(num_bones):
            w_mat[bi] = arrs.get(f"b_{bi}", np.zeros(n_v, dtype=np.float32))

        e0, e1 = mesh_edge_arrays(mesh)
        comp_labels = label_components(n_v, e0, e1)

        adj = [[] for _ in range(n_v)]
        for u_e, v_e in zip(e0.tolist(), e1.tolist()):
            adj[u_e].append(v_e)
            adj[v_e].append(u_e)

        comp_tris_map = {}
        for lt in mesh.loop_triangles:
            t0, t1, t2 = int(lt.vertices[0]), int(lt.vertices[1]), int(lt.vertices[2])
            cid = int(comp_labels[t0])
            comp_tris_map.setdefault(cid, []).append((t0, t1, t2))

        max_rad = max(bone_radii) * 1.65
        in_region_mask = (near_bone_dist <= max_rad) | (w_mat.sum(axis=0) > 0.0005)

        obj_states.append({
            "obj": obj,
            "world": world,
            "out_dirs": out_dirs,
            "rigid_mask": rigid_mask,
            "w_mat": w_mat,
            "w_orig": w_mat.copy(),
            "comp_labels": comp_labels,
            "adj": adj,
            "in_region_mask": in_region_mask,
        })

        uniq_c = np.unique(comp_labels)
        for cid in uniq_c.tolist():
            cid = int(cid)
            c_mask = (comp_labels == cid)
            if float(np.mean(rigid_mask[c_mask])) > 0.5:
                continue
            active_c_vis = np.nonzero(c_mask & (w_mat.sum(axis=0) > 0.0005) & ~rigid_mask)[0]
            if len(active_c_vis) < 3:
                continue
            tris_c = comp_tris_map.get(cid, [])
            if not tris_c:
                continue
            c_vis = np.nonzero(c_mask)[0]
            global_to_local = {int(gv): li for li, gv in enumerate(c_vis.tolist())}
            local_verts = [Vector(world[gv]) for gv in c_vis.tolist()]
            local_tris = []
            valid_global_tris = []
            for t0, t1, t2 in tris_c:
                if not (in_region_mask[t0] or in_region_mask[t1] or in_region_mask[t2]):
                    continue
                if t0 in global_to_local and t1 in global_to_local and t2 in global_to_local:
                    local_tris.append((global_to_local[t0], global_to_local[t1], global_to_local[t2]))
                    valid_global_tris.append((t0, t1, t2))
            if not local_tris:
                continue
            bvh = BVHTree.FromPolygons(local_verts, local_tris)
            cand_shells[(o_idx, cid)] = {
                "o_idx": o_idx,
                "cid": cid,
                "c_vis": c_vis,
                "active_c_vis": active_c_vis,
                "bvh": bvh,
                "local_tris": local_tris,
                "global_tris": valid_global_tris,
            }

    if len(cand_shells) < 2:
        return 0

    # 컴포넌트 쌍별 외피(Outer Shell) 피복 관계 판별 (삼각형 면 법선 norm 및 본 외향 벡터 out_dirs 기준)
    shell_keys = list(cand_shells.keys())
    outer_of = {k: set() for k in shell_keys}
    for key_i in shell_keys:
        sh_i = cand_shells[key_i]
        st_i = obj_states[sh_i["o_idx"]]
        world_i = st_i["world"]
        out_dirs_i = st_i["out_dirs"]
        act_vis_i = sh_i["active_c_vis"]

        for key_j in shell_keys:
            if key_i == key_j:
                continue
            bvh_j = cand_shells[key_j]["bvh"]
            ds_list = []
            for vi in act_vis_i.tolist():
                p_i = Vector(world_i[vi])
                loc, norm, tidx, dist = bvh_j.find_nearest(p_i, 0.008)
                if loc is not None and norm is not None and 0.00025 <= dist <= 0.008:
                    n_vec = Vector(norm)
                    if n_vec.dot(out_dirs_i[vi]) < 0.0:
                        n_vec = -n_vec
                    ds = (loc - p_i).dot(n_vec)
                    ds_list.append(ds)
            min_req = max(6, int(len(act_vis_i) * 0.12))
            if len(ds_list) >= min_req:
                arr = np.array(ds_list, dtype=np.float64)
                if float(np.mean(arr > 0.0003)) >= 0.75 and float(np.median(arr)) >= 0.0005:
                    outer_of[key_i].add(key_j)

    # 외곽 쉘 -> 내부 쉘 위상 정렬 순서로 무게중심 가중치 등각 투영
    ordered_keys = sorted(shell_keys, key=lambda k: len(outer_of[k]))
    total_adjusted = 0

    for key_i in ordered_keys:
        outers = outer_of[key_i]
        if not outers:
            continue
        sh_i = cand_shells[key_i]
        st_i = obj_states[sh_i["o_idx"]]
        world_i = st_i["world"]
        out_dirs_i = st_i["out_dirs"]
        w_mat_i = st_i["w_mat"]

        proj_map_i = {}
        for vi in sh_i["active_c_vis"].tolist():
            p_i = Vector(world_i[vi])
            od_i = out_dirs_i[vi]

            best_hit = None
            best_dist = 1e9
            for key_j in outers:
                sh_j = cand_shells[key_j]
                loc, norm, tidx, dist = sh_j["bvh"].find_nearest(p_i, search_radius)
                if loc is not None and norm is not None and tidx is not None and dist < best_dist:
                    n_vec = Vector(norm)
                    if n_vec.dot(od_i) < 0.0:
                        n_vec = -n_vec
                    dvec = loc - p_i
                    d_norm = dvec.dot(n_vec)
                    d_tang = (dvec - d_norm * n_vec).length
                    if d_norm >= -0.0005 and d_tang <= 0.008:
                        best_dist = dist
                        best_hit = (sh_j, loc, int(tidx), d_norm, d_tang)

            if best_hit is not None:
                sh_j, loc, tidx, d_norm, d_tang = best_hit
                st_j = obj_states[sh_j["o_idx"]]
                world_j = st_j["world"]
                w_mat_j = st_j["w_mat"]
                t0, t1, t2 = sh_j["global_tris"][tidx]
                bu, bv, bw = _barycentric_weights_3d(
                    loc, Vector(world_j[t0]), Vector(world_j[t1]), Vector(world_j[t2])
                )

                alpha = 1.0 if d_tang <= 0.0025 else max(0.0, (0.008 - d_tang) / 0.0055)
                proj_map_i[vi] = (sh_j, t0, t1, t2, bu, bv, bw, alpha, float(w_mat_i[0, vi]) if num_bones > 0 else 0.0)
                for bi in range(num_bones):
                    w_tri = float(bu * w_mat_j[bi, t0] + bv * w_mat_j[bi, t1] + bw * w_mat_j[bi, t2])
                    w_mat_i[bi, vi] = alpha * w_tri + (1.0 - alpha) * float(w_mat_i[bi, vi])
                total_adjusted += 1
        sh_i["proj_map"] = proj_map_i

    # 오브젝트 내부 동일 좌표(UV 심/하드 엣지) 그룹 사전 구축 (인메모리 동기화용)
    for st in obj_states:
        world = st["world"]
        n_v = len(world)
        kd_self = KDTree(n_v)
        for vi in range(n_v):
            kd_self.insert(Vector(world[vi]), vi)
        kd_self.balance()
        coincident_groups = []
        coincident_map = {}
        visited = np.zeros(n_v, dtype=bool)
        for vi in range(n_v):
            if visited[vi]:
                continue
            hits = kd_self.find_range(Vector(world[vi]), 0.0002)
            grp = [h[1] for h in hits]
            for g_vi in grp:
                visited[g_vi] = True
                coincident_map[g_vi] = grp
            if len(grp) > 1:
                coincident_groups.append(grp)
        st["coincident_groups"] = coincident_groups
        st["coincident_map"] = coincident_map

    def _sync_in_memory_coincident(st):
        w_m = st["w_mat"]
        rig_m = st["rigid_mask"]
        for grp in st["coincident_groups"]:
            rig_vs = [v for v in grp if rig_m[v]]
            for bi in range(num_bones):
                if rig_vs:
                    val = float(w_m[bi, rig_vs[0]])
                else:
                    val = float(np.mean([w_m[bi, v] for v in grp]))
                for v in grp:
                    if not rig_m[v]:
                        w_m[bi, v] = val

    for st in obj_states:
        _sync_in_memory_coincident(st)

    # [다층 삼각망 변형 교차 검증 및 국소 등각 평탄화 (Deformed BVH Overlap Elimination)]:
    # 0.3mm ~ 1.5mm 초밀착 경계(예: 밑가슴/가슴골 의상 경계)에서 내피와 외피의 삼각 분할(Triangulation) 방향이 다르면,
    # 정점 무게중심 투영만으로는 급격한 가중치 경사 구간에서 모서리 대각선 교차가 발생할 수 있습니다.
    # 6방향 극한 변위 시뮬레이션으로 실제 교차하는 삼각형 쌍을 검출하여 외피 국소 경사를 완만화하고 재투영합니다.
    test_motions = [
        [Vector((0.0, -0.035, -0.025)) for _ in range(num_bones)],
        [Vector((0.0, 0.035, -0.025)) for _ in range(num_bones)],
        [Vector((0.0, 0.0, -0.040)) for _ in range(num_bones)],
        [Vector((0.0, -0.035, 0.035)) for _ in range(num_bones)],
        [Vector((0.0, 0.035, 0.035)) for _ in range(num_bones)],
        [Vector((0.0, -0.030, 0.015)) for _ in range(num_bones)],
        [Vector((0.0, 0.030, 0.015)) for _ in range(num_bones)],
        [
            Vector((0.015, 0.030, -0.030)) if (bi % 2 == 0) else Vector((-0.015, -0.020, 0.030))
            for bi in range(num_bones)
        ],
        [
            Vector((0.015, -0.030, -0.030)) if (bi % 2 == 0) else Vector((-0.015, 0.020, 0.030))
            for bi in range(num_bones)
        ],
    ]

    rest_overlaps = {}
    for key_i in ordered_keys:
        for key_j in outer_of[key_i]:
            sh_i = cand_shells[key_i]
            sh_j = cand_shells[key_j]
            rest_overlaps[(key_i, key_j)] = set(sh_j["bvh"].overlap(sh_i["bvh"]))

    for _iter in range(20):
        any_overlap_fixed = False
        for key_i in ordered_keys:
            outers = outer_of[key_i]
            if not outers:
                continue
            sh_i = cand_shells[key_i]
            st_i = obj_states[sh_i["o_idx"]]
            world_i = st_i["world"]
            w_mat_i = st_i["w_mat"]
            rig_i = st_i["rigid_mask"]
            c_vis_i = sh_i["c_vis"].tolist()
            local_tris_i = sh_i["local_tris"]
            global_tris_i = sh_i["global_tris"]
            proj_map_i = sh_i.get("proj_map", {})

            for key_j in outers:
                sh_j = cand_shells[key_j]
                st_j = obj_states[sh_j["o_idx"]]
                world_j = st_j["world"]
                w_mat_j = st_j["w_mat"]
                rig_j = st_j["rigid_mask"]
                c_vis_j = sh_j["c_vis"].tolist()
                local_tris_j = sh_j["local_tris"]
                global_tris_j = sh_j["global_tris"]
                rest_ov = rest_overlaps.get((key_i, key_j), set())

                bad_pairs = set()
                for disp_vecs in test_motions:
                    verts_i_def = []
                    for gv in c_vis_i:
                        p = Vector(world_i[gv])
                        for bi in range(num_bones):
                            w_val = float(w_mat_i[bi, gv])
                            if w_val > 1e-5:
                                p = p + disp_vecs[bi] * w_val
                        verts_i_def.append(p)

                    verts_j_def = []
                    for gv in c_vis_j:
                        p = Vector(world_j[gv])
                        for bi in range(num_bones):
                            w_val = float(w_mat_j[bi, gv])
                            if w_val > 1e-5:
                                p = p + disp_vecs[bi] * w_val
                        verts_j_def.append(p)

                    bvh_i_def = BVHTree.FromPolygons(verts_i_def, local_tris_i)
                    bvh_j_def = BVHTree.FromPolygons(verts_j_def, local_tris_j)
                    ov = set(bvh_j_def.overlap(bvh_i_def)) - rest_ov
                    if ov:
                        bad_pairs.update(ov)

                if not bad_pairs:
                    continue

                any_overlap_fixed = True
                # 1. 외피(sh_j) 변경 전 각 내피 정점의 투영 기준값(prev_w_tri) 기록
                prev_proj = {}
                for vi, (p_sh_j, t0, t1, t2, bu, bv, bw, alpha, _) in proj_map_i.items():
                    if p_sh_j is sh_j and not rig_i[vi]:
                        prev_proj[vi] = [
                            float(bu * w_mat_j[bi, t0] + bv * w_mat_j[bi, t1] + bw * w_mat_j[bi, t2])
                            for bi in range(num_bones)
                        ]

                # 2. 교차 삼각형 쌍의 외피(c0,c1,c2) 국소 구배 완화 (동일 좌표 UV 심 쌍둥이 정점 일괄 포함)
                cmap_j = st_j["coincident_map"]
                cmap_i = st_i["coincident_map"]
                c_accum = {}
                for tj_idx, ti_idx in bad_pairs:
                    c0, c1, c2 = global_tris_j[tj_idx]
                    s0, s1, s2 = global_tris_i[ti_idx]
                    tw_vec = [
                        float(
                            w_mat_j[bi, c0] + w_mat_j[bi, c1] + w_mat_j[bi, c2]
                            + w_mat_i[bi, s0] + w_mat_i[bi, s1] + w_mat_i[bi, s2]
                        ) / 6.0
                        for bi in range(num_bones)
                    ]
                    for cv in (c0, c1, c2):
                        for cv_twin in cmap_j.get(cv, [cv]):
                            if not rig_j[cv_twin]:
                                c_accum.setdefault(cv_twin, []).append(tw_vec)

                for cv, tw_list in c_accum.items():
                    n_h = float(len(tw_list))
                    for bi in range(num_bones):
                        tw = sum(v[bi] for v in tw_list) / n_h
                        w_mat_j[bi, cv] = 0.50 * tw + 0.50 * float(w_mat_j[bi, cv])

                _sync_in_memory_coincident(st_j)

                # 3. 외피 변화량(Delta Projection)만큼 내피에 등각 반영 (이전 반복의 내피 보정치를 지우지 않고 보존)
                for vi, (p_sh_j, t0, t1, t2, bu, bv, bw, alpha, _) in proj_map_i.items():
                    if vi in prev_proj:
                        old_vec = prev_proj[vi]
                        for bi in range(num_bones):
                            new_w_tri = float(bu * w_mat_j[bi, t0] + bv * w_mat_j[bi, t1] + bw * w_mat_j[bi, t2])
                            w_mat_i[bi, vi] = max(0.0, float(w_mat_i[bi, vi]) + alpha * (new_w_tri - old_vec[bi]))

                # 4. 교차 삼각형 쌍의 내피 정점(s0,s1,s2) 및 UV 심 쌍둥이 정점을 충돌 외피 삼각형 평균으로 수렴
                s_accum = {}
                for tj_idx, ti_idx in bad_pairs:
                    c0, c1, c2 = global_tris_j[tj_idx]
                    s0, s1, s2 = global_tris_i[ti_idx]
                    c_mean_vec = [
                        float(w_mat_j[bi, c0] + w_mat_j[bi, c1] + w_mat_j[bi, c2]) / 3.0
                        for bi in range(num_bones)
                    ]
                    for sv in (s0, s1, s2):
                        for sv_twin in cmap_i.get(sv, [sv]):
                            if not rig_i[sv_twin]:
                                s_accum.setdefault(sv_twin, []).append(c_mean_vec)

                for sv, cm_list in s_accum.items():
                    n_h = float(len(cm_list))
                    for bi in range(num_bones):
                        cm = sum(v[bi] for v in cm_list) / n_h
                        w_mat_i[bi, sv] = 0.70 * cm + 0.30 * float(w_mat_i[bi, sv])
                    total_adjusted += 1

                _sync_in_memory_coincident(st_i)

        if not any_overlap_fixed:
            break

    # 변경된 가중치를 버텍스 그룹에 일괄 반영
    for st in obj_states:
        obj = st["obj"]
        w_mat = st["w_mat"]
        w_orig = st["w_orig"]
        rigid_mask = st["rigid_mask"]
        n_v = len(st["world"])

        for bi, vgn in enumerate(vg_names):
            vg = obj.vertex_groups.get(vgn) or obj.vertex_groups.new(name=vgn)
            diff_vis = np.nonzero((np.abs(w_mat[bi] - w_orig[bi]) > 1e-4) & ~rigid_mask)[0]
            for vi in diff_vis.tolist():
                new_w = float(w_mat[bi, vi])
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

    slot_rigid_by_obj = {}
    for item in getattr(props, "rigid_items", []):
        rdata = get_rigid_data(item)
        if rdata:
            for oname, vis in rdata.items():
                if vis:
                    slot_rigid_by_obj.setdefault(oname, set()).update(vis)
    has_slot_rigid = bool(slot_rigid_by_obj)

    obj_data = {}
    for obj in target_objs:
        ensure_object_visible(context, obj)

        vg_rigid = obj.vertex_groups.get(VG_RIGID)
        world = world_vertex_positions(obj)
        n_verts = len(world)

        rigid_mask = np.zeros(n_verts, dtype=bool)
        if has_slot_rigid:
            reg_r_vis = [v for v in sorted(slot_rigid_by_obj.get(obj.name, set())) if 0 <= v < n_verts]
            if vg_rigid:
                vg_rigid.remove(list(range(n_verts)))
            if reg_r_vis:
                if not vg_rigid:
                    vg_rigid = obj.vertex_groups.new(name=VG_RIGID)
                vg_rigid.add(reg_r_vis, 1.0, 'REPLACE')
                rigid_mask[reg_r_vis] = True
        else:
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
                        if (n_rig / cnt) >= 0.50:
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

    # 3.0. 활성 본 메타데이터 수집 및 오브젝트별 통합 물리 허용 풀(Union Pool) 구축
    resolved_bones = {}
    slot_verts_by_bone = {}
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
        b_item = get_bone_item(props, bname)
        b_type = getattr(b_item, "bone_type", "MORPH") if b_item else "MORPH"
        resolved_bones[bname] = {
            "entry": bentry,
            "params": get_bone_params(props, bname),
            "item": b_item,
            "bone_type": b_type,
        }
        slot_verts_by_bone[bname] = get_bone_data(b_item) if b_item else {}

    for obj, odata in obj_data.items():
        n_v = len(odata["world"])
        rigid_mask = odata["rigid_mask"]
        e0, e1 = mesh_edge_arrays(obj.data)

        per_bone_masks = {}
        union_mask = np.zeros(n_v, dtype=bool)
        for bname in resolved_bones.keys():
            vis = slot_verts_by_bone.get(bname, {}).get(obj.name, [])
            if vis:
                bm = np.zeros(n_v, dtype=bool)
                valid_vis = [v for v in vis if v < n_v]
                if valid_vis:
                    bm[valid_vis] = True
                    per_bone_masks[bname] = bm
                    union_mask |= bm

        # 인접 슬롯 간 선택 경계에 1~2엣지 미세 틈이 있을 경우 외곽은 넓히지 않고 내부 사잇길만 자동 연결
        if len(per_bone_masks) >= 2 and len(e0) > 0:
            dilated_map = {
                bn: dilate_mask_topological(bm, e0, e1, steps=2)
                for bn, bm in per_bone_masks.items()
            }
            bn_list = list(dilated_map.keys())
            for i in range(len(bn_list)):
                for j in range(i + 1, len(bn_list)):
                    union_mask |= (dilated_map[bn_list[i]] & dilated_map[bn_list[j]])

        union_mask &= ~rigid_mask
        odata["union_mask"] = union_mask
        odata["has_registered_union"] = bool(np.any(union_mask))
        odata["per_bone_masks"] = per_bone_masks

    # 3.1. 인접/대칭 모핑 본 쌍(예: 좌/우 가슴) 자동 감지
    paired_bones = set()
    bridge_pairs = []
    morph_bnames = [bn for bn, info in resolved_bones.items() if info["bone_type"] == "MORPH"]
    for i in range(len(morph_bnames)):
        bn_a = morph_bnames[i]
        if bn_a in paired_bones:
            continue
        ea = resolved_bones[bn_a]["entry"]
        ha = np.array(ea["head"], dtype=np.float64)
        ra = max(0.02, float(ea["radius"]))

        best_partner = None
        best_dist = 1e9
        for j in range(i + 1, len(morph_bnames)):
            bn_b = morph_bnames[j]
            if bn_b in paired_bones:
                continue
            eb = resolved_bones[bn_b]["entry"]
            hb = np.array(eb["head"], dtype=np.float64)
            rb = max(0.02, float(eb["radius"]))

            d_ab = float(np.linalg.norm(hb - ha))
            dz = abs(float(ha[2] - hb[2]))
            if 0.015 < d_ab <= (ra + rb) * 2.5 and dz < 0.12 and d_ab < best_dist:
                best_dist = d_ab
                best_partner = bn_b

        if best_partner is not None:
            paired_bones.add(bn_a)
            paired_bones.add(best_partner)
            bridge_pairs.append((bn_a, best_partner))

    # 3.2. 인접 모핑 본 쌍에 대한 통합 풀(Union Pool) & 브릿지 크로스 블렌딩 가중치 계산
    for bn_a, bn_b in bridge_pairs:
        info_a = resolved_bones[bn_a]
        info_b = resolved_bones[bn_b]
        ea, pa = info_a["entry"], info_a["params"]
        eb, pb = info_b["entry"], info_b["params"]

        ha = np.array(ea["head"], dtype=np.float64)
        ta = np.array(ea["tail"], dtype=np.float64)
        ca = np.array(ea["centroid"], dtype=np.float64)
        fa = np.array(ea["fwd_unit"], dtype=np.float64)
        ra = max(0.02, float(ea["radius"]))

        hb = np.array(eb["head"], dtype=np.float64)
        tb = np.array(eb["tail"], dtype=np.float64)
        cb = np.array(eb["centroid"], dtype=np.float64)
        fb = np.array(eb["fwd_unit"], dtype=np.float64)
        rb = max(0.02, float(eb["radius"]))

        vg_name_a = f"XXMI_{bn_a}"
        vg_name_b = f"XXMI_{bn_b}"
        lock_reg = pa["lock_registered_verts"] and pb["lock_registered_verts"]

        pair_cand = []
        for obj, odata in obj_data.items():
            vg_a = obj.vertex_groups.get(vg_name_a) or obj.vertex_groups.new(name=vg_name_a)
            vg_b = obj.vertex_groups.get(vg_name_b) or obj.vertex_groups.new(name=vg_name_b)
            world = odata["world"]
            rigid_mask = odata["rigid_mask"]
            n_v = len(world)

            # 비강체 정점들의 기존 가중치 초기화 (반경 축소 시 잔여 가중치 완전 제거)
            non_rig_list = np.nonzero(~rigid_mask)[0].tolist()
            if non_rig_list:
                vg_a.remove(non_rig_list)
                vg_b.remove(non_rig_list)

            if odata["has_registered_union"] and lock_reg:
                cand_mask = odata["union_mask"]
            else:
                dist_a = np.linalg.norm(world - ca, axis=1)
                front_a = ((world - ha) @ fa) >= -ra * 0.25
                dist_b = np.linalg.norm(world - cb, axis=1)
                front_b = ((world - hb) @ fb) >= -rb * 0.25
                env_mask = ((dist_a <= ra * 1.30) & front_a) | ((dist_b <= rb * 1.30) & front_b)
                if odata["has_registered_union"]:
                    cand_mask = (odata["union_mask"] | env_mask) & ~rigid_mask
                else:
                    cand_mask = env_mask & ~rigid_mask

            cand_idx = np.nonzero(cand_mask)[0]
            for vi in cand_idx.tolist():
                pair_cand.append((obj, vg_a, vg_b, vi, world[vi]))

        if not pair_cand:
            continue

        cand_pts = np.array([c[4] for c in pair_cand], dtype=np.float64)
        w_a_arr, w_b_arr = calculate_bridged_pair_weights(
            cand_pts,
            ha, ta, ra, pa["falloff_power"], pa["peak_weight"],
            hb, tb, rb, pb["falloff_power"], pb["peak_weight"],
        )

        for (obj, vg_a, vg_b, vi, _), wa, wb in zip(pair_cand, w_a_arr.tolist(), w_b_arr.tolist()):
            if wa > 0.0005:
                vg_a.add([vi], max(0.0, min(1.0, wa)), 'REPLACE')
                total_recomputed_verts += 1
            else:
                vg_a.remove([vi])

            if wb > 0.0005:
                vg_b.add([vi], max(0.0, min(1.0, wb)), 'REPLACE')
                total_recomputed_verts += 1
            else:
                vg_b.remove([vi])

    # 3.3. 단독 물리 본별 가중치 계산
    for bname, info in resolved_bones.items():
        if bname in paired_bones:
            continue

        bentry = info["entry"]
        b_params = info["params"]
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
        b_item_verts = slot_verts_by_bone.get(bname, {})

        side_cand = []
        for obj, odata in obj_data.items():
            vg = obj.vertex_groups.get(vg_target_name) or obj.vertex_groups.new(name=vg_target_name)
            world = odata["world"]
            rigid_mask = odata["rigid_mask"]
            n_v = len(world)

            non_rig_list = np.nonzero(~rigid_mask)[0].tolist()
            if non_rig_list:
                vg.remove(non_rig_list)

            slot_obj_verts = set(b_item_verts.get(obj.name, []))
            if slot_obj_verts and side_lock_registered:
                cand_mask = odata["union_mask"] if odata["has_registered_union"] else np.zeros(n_v, dtype=bool)
            else:
                dist = np.linalg.norm(world - cent, axis=1)
                in_radius = (dist <= eff_radius * 1.30)
                dot_fwd = (world - head) @ fwd_u
                front_ok = (dot_fwd >= -eff_radius * 0.25)
                cand_mask = (in_radius & front_ok) & ~rigid_mask
                if slot_obj_verts:
                    for vi in slot_obj_verts:
                        if vi < n_v and not rigid_mask[vi]:
                            cand_mask[vi] = True

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

    # 4. 라플라시안 스무딩 (통합 허용 풀 내부에서는 중앙 경계 제약 없이 부드럽게 블렌딩)
    for bname in active_bones:
        b_params = get_bone_params(props, bname)
        s_iters = b_params["smooth_iterations"]
        if s_iters > 0:
            vg_name = f"XXMI_{bname}"
            for obj, odata in obj_data.items():
                vg = obj.vertex_groups.get(vg_name)
                if vg:
                    n_v = len(odata["world"])
                    if odata.get("has_registered_union", False):
                        allowed_m = odata["union_mask"].copy()
                    else:
                        allowed_m = np.zeros(n_v, dtype=bool)
                    cur_arr = read_vertex_group_arrays(obj, {vg.index: "w"}).get("w", np.zeros(n_v))
                    allowed_m |= (cur_arr > 0.0001)

                    smooth_mesh_weights(
                        obj, [vg], odata["rigid_mask"],
                        iters=s_iters, alpha=0.45, lock_registered=lock_registered,
                        allowed_mask=allowed_m
                    )

    target_peak_cap = float(peak_w)
    active_vg_names = [f"XXMI_{bn}" for bn in active_bones]

    # 5. 강체 표면 밀착 가중치 동기화 (단일 통합 소프트 표면 KD-Tree 기반)
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

    # 모든 활성 본의 가중치 벡터를 포함한 단일 통합 연질(Soft) 표면 KD-Tree 구축
    # (본별 개별 KD-Tree 사용 시 좌/우 본에 걸친 단일 강체 슬롯이 양쪽 피크치를 각각 평균내어 합이 2배로 뛰는 현상 원천 차단)
    soft_pts = []
    soft_weight_vecs = []
    for obj, odata in obj_data.items():
        world = odata["world"]
        rigid_mask = odata["rigid_mask"]
        n_v = len(world)
        gmap = {}
        for bi, bname in enumerate(active_bones):
            vg = obj.vertex_groups.get(f"XXMI_{bname}")
            if vg:
                gmap[vg.index] = bname
        if not gmap:
            continue
        cur_arrs = read_vertex_group_arrays(obj, gmap)
        w_cols = {bn: cur_arrs.get(bn, np.zeros(n_v, dtype=np.float32)) for bn in active_bones}
        for vi in range(n_v):
            if rigid_mask[vi]:
                continue
            total_w_vi = sum(float(w_cols[bn][vi]) for bn in active_bones)
            if total_w_vi > 0.001:
                soft_pts.append(Vector(world[vi]))
                soft_weight_vecs.append({bn: float(w_cols[bn][vi]) for bn in active_bones})

    soft_kd = None
    if soft_pts:
        soft_kd = KDTree(len(soft_pts))
        for k, p in enumerate(soft_pts):
            soft_kd.insert(p, k)
        soft_kd.balance()

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

        if soft_kd is not None:
            dists_and_vecs = []
            for pt in all_pts_slot:
                res = soft_kd.find(pt)
                if res is not None and res[0] is not None:
                    dists_and_vecs.append((res[2], soft_weight_vecs[res[1]]))

            if dists_and_vecs:
                min_d = min(d for d, _ in dists_and_vecs)
                if min_d < 0.08:
                    contact_pts = [(d, wvec) for d, wvec in dists_and_vecs if d <= min_d + 0.015]
                    inv_d_sum = sum(1.0 / max(d, 0.002) for d, _ in contact_pts)
                    dist_factor = max(0.0, 1.0 - max(0.0, min_d - 0.04) / 0.04)
                    for bn in active_bones:
                        base_w = sum(wvec[bn] * (1.0 / max(d, 0.002)) for d, wvec in contact_pts) / max(1e-6, inv_d_sum)
                        slot_unified_weights[bn] = base_w * dist_factor * w_scale

                    sum_slot_w = sum(slot_unified_weights.values())
                    max_allowed_slot = target_peak_cap * w_scale
                    if sum_slot_w > max_allowed_slot and sum_slot_w > 1e-6:
                        sc = max_allowed_slot / sum_slot_w
                        for bn in active_bones:
                            slot_unified_weights[bn] *= sc

        # 3. 강체 슬롯에 속한 모든 오브젝트(예: Hair의 무전기 본체, Body의 무전기 케이스)의
        #    모든 정점에 100.000% 완벽히 동일한 단일 가중치 부여 (상대 변위 0 -> 뚫림 원천 차단)
        for robj, vis, _ in slot_objs_data:
            vg_rigid = robj.vertex_groups.get(VG_RIGID) or robj.vertex_groups.new(name=VG_RIGID)
            vg_rigid.add(vis, 1.0, 'REPLACE')
            if robj in obj_data:
                valid_vis = [v for v in vis if v < len(obj_data[robj]["rigid_mask"])]
                if valid_vis:
                    obj_data[robj]["rigid_mask"][valid_vis] = True

            for bn in active_bones:
                vg_target = robj.vertex_groups.get(f"XXMI_{bn}") or robj.vertex_groups.new(name=f"XXMI_{bn}")
                slot_w = float(slot_unified_weights.get(bn, 0.0))
                if slot_w > 0.001:
                    vg_target.add(vis, min(1.0, slot_w), 'REPLACE')
                else:
                    vg_target.remove(vis)

        applied_slots += 1

    # 5.5. 강체와 동일 좌표를 공유하는 연질 봉합선(Seam) 정점 선행 동기화 및 잠금
    #      (테이퍼 단계에서 봉합선 인접 연질 폴리곤들이 부드럽게 이어지도록 유도하여 칼라/버클 경계 찢어짐 원천 방지)
    seam_locked_by_obj = {obj: np.zeros(len(odata["world"]), dtype=bool) for obj, odata in obj_data.items()}
    rigid_seam_pts = []
    rigid_seam_wvecs = []
    for obj, odata in obj_data.items():
        world = odata["world"]
        rigid_mask = odata["rigid_mask"]
        if not np.any(rigid_mask):
            continue
        n_v = len(world)
        gmap = {}
        for bname in active_bones:
            vg = obj.vertex_groups.get(f"XXMI_{bname}")
            if vg:
                gmap[vg.index] = bname
        cur_arrs = read_vertex_group_arrays(obj, gmap)
        w_cols = {bn: cur_arrs.get(bn, np.zeros(n_v, dtype=np.float32)) for bn in active_bones}
        for vi in np.nonzero(rigid_mask)[0].tolist():
            rigid_seam_pts.append(Vector(world[vi]))
            rigid_seam_wvecs.append({bn: float(w_cols[bn][vi]) for bn in active_bones})

    if rigid_seam_pts:
        kd_rseam = KDTree(len(rigid_seam_pts))
        for k, p in enumerate(rigid_seam_pts):
            kd_rseam.insert(p, k)
        kd_rseam.balance()

        for obj, odata in obj_data.items():
            world = odata["world"]
            rigid_mask = odata["rigid_mask"]
            seam_locked = seam_locked_by_obj[obj]
            for vi in range(len(world)):
                if rigid_mask[vi]:
                    continue
                res = kd_rseam.find(Vector(world[vi]))
                if res is not None and res[0] is not None and res[2] <= 0.0002:
                    seam_locked[vi] = True
                    wvec = rigid_seam_wvecs[res[1]]
                    for bn in active_bones:
                        vg_t = obj.vertex_groups.get(f"XXMI_{bn}") or obj.vertex_groups.new(name=f"XXMI_{bn}")
                        rw = float(wvec[bn])
                        if rw > 0.0005:
                            vg_t.add([vi], min(1.0, rw), 'REPLACE')
                        else:
                            vg_t.remove([vi])

    # 6. 경계면 양방향 완만화(Boundary Gradient Taper)로 외곽 및 강체 봉합선 절벽 찢어짐 원천 차단
    smooth_boundary_gradient_taper(
        target_objs, active_vg_names, max_delta=0.07, passes=5, seam_locked_by_obj=seam_locked_by_obj
    )

    # 7. 가슴 중앙선 중복 가중치 재정규화 (가슴골 2배 돌출 원천 차단)
    normalize_overlapping_bone_weights(
        target_objs, active_bones, max_total_weight=target_peak_cap
    )

    # 7.5. 외피(겉옷)의 UV 심/하드 엣지 분리 정점들을 먼저 동기화하여 등각 투영 시 삼각형 내부 단차 제거
    synchronize_coincident_vertices_across_objects(
        target_objs, vg_names=active_vg_names, tol=0.0002
    )

    # 8. 다층 쉘 무게중심 등각 동기화 (Conformal Multi-Shell Synchronization):
    #    경계면 테이퍼까지 완료된 최종 외피(겉옷) 삼각망의 가중치 장을 내피(속살) 정점에 무게중심 투영하여
    #    전진/후진/상하 모든 진동 위상에서 다층 메쉬 관통을 0으로 원천 봉쇄합니다.
    enforce_radial_monotonic_non_penetration(
        target_objs, bone_info, active_bones, search_radius=0.014
    )

    # 9. [절대 최종 단계]: 모든 동일 위치 공유 정점(UV 심/하드 엣지) 100% 완전 동기화
    num_synced = synchronize_coincident_vertices_across_objects(
        target_objs, vg_names=active_vg_names, tol=0.0002
    )

    return updated_obj_count, f"{updated_obj_count}개 메쉬 ({total_recomputed_verts}개 정점, 동기화 {num_synced}개, 강체슬롯 {applied_slots}개) 가중치 갱신 완료"


