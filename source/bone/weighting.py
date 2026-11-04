import math
import numpy as np
from collections import deque
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import (
    ensure_object_visible,
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
from .data import get_bone_params

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

    # 1. 전방 깊이 축 투영 거리 및 매끄러운 돔 감쇠
    t_proj = pts_rel @ bone_u
    t_norm = np.clip(t_proj / bone_len, 0.0, 1.0)
    w_depth = np.sin(np.sqrt(t_norm) * (np.pi * 0.5))

    # 2. 방사형 수직 거리 및 Hann Cosine 감쇠
    v_perp = pts_rel - np.outer(t_proj, bone_u)
    rho = np.linalg.norm(v_perp, axis=1)
    radius = max(0.02, float(eff_radius))
    rho_norm = np.clip(rho / radius, 0.0, 1.0)
    w_radial = 0.5 * (1.0 + np.cos(np.pi * rho_norm))

    # 3. 해부학적 합성 가중치 계산
    w_comb = peak_w * np.power(np.maximum(0.0, w_depth * w_radial), falloff_p)
    w_arr = np.clip(w_comb, 0.005, 1.0)

    # 4. UV 심 및 분리 정점 동기화
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

    def _eval_side_weight(pt, side):
        if side not in bone_info:
            return 0.0
        head = Vector(bone_info[side]["head"])
        tail = Vector(bone_info[side]["tail"])
        bone_vec = tail - head
        bone_len = bone_vec.length
        fwd_u = bone_vec.normalized() if bone_len > 1e-4 else Vector((0.0, -1.0, 0.0))
        radius = max(0.02, bone_info[side]["radius"])

        v = pt - head
        t = v.dot(fwd_u)
        t_norm = max(0.0, min(1.0, t / max(1e-4, bone_len)))
        w_depth = math.sin(math.sqrt(t_norm) * (math.pi * 0.5))

        v_perp = v - t * fwd_u
        rho = v_perp.length
        rho_norm = max(0.0, min(1.0, rho / radius))
        w_radial = 0.5 * (1.0 + math.cos(math.pi * rho_norm))

        w = peak_w * math.pow(max(0.0, w_depth * w_radial), falloff_p)
        return max(0.05, min(1.0, w)) if w > 0.0 else 0.0

    for comp in clusters.values():
        all_pts = np.concatenate([isl["pts"] for isl in comp], axis=0)
        c_cent = Vector(all_pts.mean(axis=0).tolist())
        min_x = float(all_pts[:, 0].min())
        max_x = float(all_pts[:, 0].max())

        has_l = "L" in bone_info
        has_r = "R" in bone_info

        if min_x < -0.004 and max_x > 0.004 and has_l and has_r:
            span = max(0.02, max_x - min_x)
            t_l = max(0.0, min(1.0, 0.5 + c_cent.x / span))
            cluster_wl = _eval_side_weight(c_cent, "L") * t_l
            cluster_wr = _eval_side_weight(c_cent, "R") * (1.0 - t_l)
        elif c_cent.x >= 0.0:
            cluster_wl = _eval_side_weight(c_cent, "L") if has_l else 0.0
            cluster_wr = 0.0
        else:
            cluster_wl = 0.0
            cluster_wr = _eval_side_weight(c_cent, "R") if has_r else 0.0

        for isl in comp:
            obj = isl["obj"]
            verts = isl["verts"]
            vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)

            if cluster_wl > 0.0:
                vg_l.add(verts, cluster_wl, 'REPLACE')
            else:
                vg_l.remove(verts)

            if cluster_wr > 0.0:
                vg_r.add(verts, cluster_wr, 'REPLACE')
            else:
                vg_r.remove(verts)

    return len(clusters)


def smooth_mesh_weights(obj, vg_l, vg_r, rigid_mask, iters=2, alpha=0.45, lock_registered=True):
    """
    메쉬 엣지 위상 구조를 기반으로 가중치를 라플라시안 스무딩합니다.
    """
    if iters <= 0 or not obj or obj.type != 'MESH':
        return
    mesh = obj.data
    e0, e1 = mesh_edge_arrays(mesh)
    if len(e0) == 0:
        return
    n_verts = len(mesh.vertices)
    group_map = {}
    if vg_l: group_map[vg_l.index] = "l"
    if vg_r: group_map[vg_r.index] = "r"
    if not group_map:
        return
    arrs = read_vertex_group_arrays(obj, group_map)

    for side_key, vg in (("l", vg_l), ("r", vg_r)):
        if side_key not in arrs or not vg:
            continue
        w = arrs[side_key].astype(np.float32)
        active_initial = (w > 0.0001) & ~rigid_mask
        if not np.any(active_initial):
            continue

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
                w[~active_initial & ~rigid_mask] = 0.0

        changed_idx = np.nonzero((w > 0.0001) & ~rigid_mask)[0]
        for vi in changed_idx.tolist():
            wi = float(w[vi])
            if wi > 0.0005:
                vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')
            else:
                vg.remove([vi])


def synchronize_coincident_vertices_across_objects(target_objs, tol=0.0012):
    """
    모든 대상 오브젝트에 걸쳐 3D 공간 상에서 동일한 위치(기본 1.2mm 이내)를 공유하는
    모든 정점(UV 심, 하드 엣지 분리점 등)의 가중치를 100% 동일하게 동기화합니다.
    """
    if not target_objs:
        return 0

    obj_entries = []

    for obj in target_objs:
        if obj.type != 'MESH':
            continue
        mesh = obj.data
        n_verts = len(mesh.vertices)
        if n_verts == 0:
            continue

        vg_l = obj.vertex_groups.get(VG_BREAST_L)
        vg_r = obj.vertex_groups.get(VG_BREAST_R)
        vg_rigid = obj.vertex_groups.get(VG_RIGID)

        group_map = {}
        if vg_l: group_map[vg_l.index] = "l"
        if vg_r: group_map[vg_r.index] = "r"
        if vg_rigid: group_map[vg_rigid.index] = "rigid"

        arrs = read_vertex_group_arrays(obj, group_map)
        world = world_vertex_positions(obj)

        zeros = np.zeros(n_verts, dtype=np.float32)
        wl = arrs.get("l", zeros)
        wr = arrs.get("r", zeros)
        rigid = (arrs.get("rigid", zeros) > 0.5)

        obj_entries.append({
            "obj": obj,
            "vg_l": vg_l,
            "vg_r": vg_r,
            "world": world,
            "wl": wl,
            "wr": wr,
            "rigid": rigid,
            "n": n_verts,
        })

    if not obj_entries:
        return 0

    active_items = []
    for o_idx, entry in enumerate(obj_entries):
        wl = entry["wl"]
        wr = entry["wr"]
        world = entry["world"]
        mask = (wl > 0.0001) | (wr > 0.0001)
        for vi in np.nonzero(mask)[0].tolist():
            active_items.append((o_idx, vi, world[vi], float(wl[vi]), float(wr[vi])))

    if not active_items:
        return 0

    kd = KDTree(len(active_items))
    for k, item in enumerate(active_items):
        kd.insert(Vector(item[2]), k)
    kd.balance()

    updates_per_obj = {o_idx: {} for o_idx in range(len(obj_entries))}
    synced_count = 0

    for o_idx, entry in enumerate(obj_entries):
        world = entry["world"]
        wl = entry["wl"]
        wr = entry["wr"]
        rigid = entry["rigid"]

        for vi in range(entry["n"]):
            if rigid[vi]:
                continue

            pt = Vector(world[vi])
            hits = kd.find_range(pt, tol)
            if not hits:
                continue

            max_wl = float(wl[vi])
            max_wr = float(wr[vi])
            for _, hit_idx, _ in hits:
                _, _, _, hw_l, hw_r = active_items[hit_idx]
                if hw_l > max_wl: max_wl = hw_l
                if hw_r > max_wr: max_wr = hw_r

            if (abs(wl[vi] - max_wl) > 1e-4) or (abs(wr[vi] - max_wr) > 1e-4):
                updates_per_obj[o_idx][vi] = (max_wl, max_wr)

    for o_idx, updates in updates_per_obj.items():
        if not updates:
            continue
        entry = obj_entries[o_idx]
        obj = entry["obj"]
        vg_l = entry["vg_l"] or obj.vertex_groups.new(name=VG_BREAST_L)
        vg_r = entry["vg_r"] or obj.vertex_groups.new(name=VG_BREAST_R)

        for vi, (new_l, new_r) in updates.items():
            if new_l > 0.0005:
                vg_l.add([vi], max(0.0, min(1.0, new_l)), 'REPLACE')
            elif new_l <= 0.0001:
                vg_l.remove([vi])

            if new_r > 0.0005:
                vg_r.add([vi], max(0.0, min(1.0, new_r)), 'REPLACE')
            elif new_r <= 0.0001:
                vg_r.remove([vi])

            synced_count += 1

    return synced_count


def rebake_mesh_weights_from_rig(context, rig, props=None):
    """
    XXMI_Breast_Rig 내의 본 위치(Breast.L / Breast.R)를 기준으로
    씬 내의 모든 관련 메쉬 오브젝트들의 버텍스 가중치를 다시 계산합니다.
    """
    if props is None:
        props = context.scene.xxmi_custom_physics_props

    bone_info = get_rig_bone_info(rig)
    if not bone_info:
        return 0, "유효한 본(Breast.L 또는 Breast.R)을 찾을 수 없습니다."

    target_objs = [
        o for o in bpy.data.objects
        if o.type == 'MESH' and (
            o.vertex_groups.get(VG_BREAST_L) or
            o.vertex_groups.get(VG_BREAST_R) or
            o.vertex_groups.get(VG_RIGID)
        )
    ]

    if not target_objs:
        target_objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not target_objs and context.active_object and context.active_object.type == 'MESH':
            target_objs = [context.active_object]
        if not target_objs:
            target_objs = [o for o in bpy.data.objects if o.type == 'MESH']

    if not target_objs:
        return 0, "씬 내에서 대상 메쉬 오브젝트를 찾을 수 없습니다."

    falloff_p = props.falloff_power
    peak_w = getattr(props, "peak_weight", 0.40)
    smooth_iters = max(
        get_bone_params(props, "Breast.L")["smooth_iterations"],
        get_bone_params(props, "Breast.R")["smooth_iterations"]
    )
    lock_registered = getattr(props, "lock_registered_verts", True)
    updated_obj_count = 0
    total_recomputed_verts = 0

    obj_data = {}
    for obj in target_objs:
        ensure_object_visible(context, obj)

        vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
        vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)
        vg_rigid = obj.vertex_groups.get(VG_RIGID)

        group_map = {vg_l.index: "l", vg_r.index: "r"}
        if vg_rigid:
            group_map[vg_rigid.index] = "rigid"
        arrs = read_vertex_group_arrays(obj, group_map)
        world = world_vertex_positions(obj)

        rigid_mask = (arrs["rigid"] > 0.5) if "rigid" in arrs else np.zeros(len(world), dtype=bool)

        n_verts = len(world)
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
            "vg_l": vg_l,
            "vg_r": vg_r,
            "arrs": arrs,
            "world": world,
            "rigid_mask": rigid_mask,
        }

    for side, prev_vg_key in (("L", "l"), ("R", "r")):
        if side not in bone_info:
            continue

        b_params = get_bone_params(props, f"Breast.{side}")
        side_falloff_p = b_params["falloff_power"]
        side_peak_w = b_params["peak_weight"]
        side_lock_registered = b_params["lock_registered_verts"]

        cent = np.array(bone_info[side]["centroid"], dtype=np.float64)
        head = np.array(bone_info[side]["head"], dtype=np.float64)
        fwd_u = np.array(bone_info[side]["fwd_unit"], dtype=np.float64)
        radius = bone_info[side]["radius"]
        eff_radius = max(0.02, radius)

        side_cand = []
        zero_entries = []

        for obj, odata in obj_data.items():
            vg = odata["vg_l"] if side == "L" else odata["vg_r"]
            world = odata["world"]
            arrs = odata["arrs"]
            rigid_mask = odata["rigid_mask"]

            prev_weighted = (arrs[prev_vg_key] > 0.0001)

            if side_lock_registered:
                cand_mask = prev_weighted & ~rigid_mask
            else:
                dist = np.linalg.norm(world - cent, axis=1)
                in_radius = (dist <= eff_radius)
                side_ok = (world[:, 0] >= -0.01) if side == "L" else (world[:, 0] <= 0.01)
                dot_fwd = (world - head) @ fwd_u
                front_ok = (dot_fwd >= -eff_radius * 0.35)
                cand_mask = (prev_weighted | (in_radius & side_ok & front_ok)) & ~rigid_mask

                non_cand_prev = np.nonzero(prev_weighted & ~cand_mask)[0]
                if len(non_cand_prev) > 0:
                    zero_entries.append((vg, non_cand_prev.tolist()))

            cand_idx = np.nonzero(cand_mask)[0]
            for vi in cand_idx.tolist():
                side_cand.append((obj, vg, vi, world[vi]))

        if not side_cand:
            continue

        cand_pts = np.array([c[3] for c in side_cand], dtype=np.float64)
        tail = np.array(bone_info[side]["tail"], dtype=np.float64)
        baked_w = calculate_anatomical_weights_with_anti_clipping(
            cand_pts, head, tail, eff_radius, side_falloff_p, side_peak_w
        )

        final_weights = baked_w.copy()

        for (obj, vg, vi, _), wi in zip(side_cand, final_weights.tolist()):
            if side_lock_registered:
                vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')
                total_recomputed_verts += 1
            else:
                if wi <= 0.0001:
                    vg.remove([vi])
                else:
                    vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')
                    total_recomputed_verts += 1

        if not side_lock_registered:
            for vg, z_list in zero_entries:
                vg.remove(z_list)

    # 중앙골 대칭 크로스 블렌딩
    center_x = 0.0
    if "L" in bone_info and "R" in bone_info:
        center_x = (bone_info["L"]["head"][0] + bone_info["R"]["head"][0]) * 0.5

    blend_w = 0.025
    for obj, odata in obj_data.items():
        vg_l = odata["vg_l"]
        vg_r = odata["vg_r"]
        world = odata["world"]
        rigid_mask = odata["rigid_mask"]

        arrs = read_vertex_group_arrays(obj, {vg_l.index: "l", vg_r.index: "r"})
        wl = arrs["l"]
        wr = arrs["r"]

        in_cleavage = (np.abs(world[:, 0] - center_x) <= blend_w) & ~rigid_mask & ((wl > 0.0001) | (wr > 0.0001))
        c_idx = np.nonzero(in_cleavage)[0]
        for vi in c_idx.tolist():
            x_rel = (world[vi, 0] - center_x) / blend_w
            t_blend = 0.5 + 0.5 * x_rel
            t_smooth = 3.0 * (t_blend ** 2) - 2.0 * (t_blend ** 3)
            sum_w = wl[vi] + wr[vi]
            new_l = sum_w * t_smooth
            new_r = sum_w * (1.0 - t_smooth)
            if new_l > 0.0005: vg_l.add([vi], max(0.0, min(1.0, float(new_l))), 'REPLACE')
            else: vg_l.remove([vi])
            if new_r > 0.0005: vg_r.add([vi], max(0.0, min(1.0, float(new_r))), 'REPLACE')
            else: vg_r.remove([vi])

        updated_obj_count += 1

    # 라플라시안 스무딩
    if smooth_iters > 0:
        for obj, odata in obj_data.items():
            smooth_mesh_weights(
                obj, odata["vg_l"], odata["vg_r"], odata["rigid_mask"],
                iters=smooth_iters, alpha=0.45, lock_registered=lock_registered
            )

    # 동일 위치 공유 정점 동기화
    num_synced = synchronize_coincident_vertices_across_objects(target_objs, tol=0.0012)

    # 레거시 / 슬롯 외 강체 정점 fallback 클러스터 통합
    slot_exclude = {}
    for item in props.rigid_items:
        rdata = get_rigid_data(item)
        if not rdata:
            continue
        for oname, vis in rdata.items():
            slot_exclude.setdefault(oname, set()).update(vis)

    num_clusters = unify_rigid_clusters_across_objects(
        target_objs, bone_info, falloff_p, peak_w, exclude_verts_by_obj=slot_exclude
    )

    # 다중 강체 슬롯 가중치 100% 동기화 적용
    applied_slots = 0
    for item in props.rigid_items:
        rdata = get_rigid_data(item)
        if not rdata:
            continue

        slot_pts = []
        for oname, vis in rdata.items():
            robj = bpy.data.objects.get(oname)
            if robj and robj.type == 'MESH':
                r_world = world_vertex_positions(robj)
                for vi in vis:
                    if vi < len(r_world):
                        slot_pts.append(r_world[vi])

        c_x = 0.0
        if slot_pts:
            c_x = float(np.mean([p[0] for p in slot_pts]))

        t_bone = item.target_bone
        if t_bone == 'BREAST_L':
            slot_wl, slot_wr = 1.0, 0.0
        elif t_bone == 'BREAST_R':
            slot_wl, slot_wr = 0.0, 1.0
        elif t_bone == 'CENTER':
            slot_wl, slot_wr = 1.0, 1.0
        else:
            if c_x > 0.012:
                slot_wl, slot_wr = 1.0, 0.0
            elif c_x < -0.012:
                slot_wl, slot_wr = 0.0, 1.0
            else:
                slot_wl, slot_wr = 1.0, 1.0

        for oname, vis in rdata.items():
            robj = bpy.data.objects.get(oname)
            if not robj or robj.type != 'MESH' or not vis:
                continue
            vg_l = robj.vertex_groups.get(VG_BREAST_L) or robj.vertex_groups.new(name=VG_BREAST_L)
            vg_r = robj.vertex_groups.get(VG_BREAST_R) or robj.vertex_groups.new(name=VG_BREAST_R)
            vg_rigid = robj.vertex_groups.get(VG_RIGID) or robj.vertex_groups.new(name=VG_RIGID)

            vg_rigid.add(vis, 1.0, 'REPLACE')
            if slot_wl > 0.0:
                vg_l.add(vis, slot_wl, 'REPLACE')
            else:
                vg_l.remove(vis)

            if slot_wr > 0.0:
                vg_r.add(vis, slot_wr, 'REPLACE')
            else:
                vg_r.remove(vis)

        applied_slots += 1

    return updated_obj_count, f"{updated_obj_count}개 메쉬 ({total_recomputed_verts}개 정점, 동기화 {num_synced}개, 강체슬롯 {applied_slots}개) 가중치 갱신 완료"

