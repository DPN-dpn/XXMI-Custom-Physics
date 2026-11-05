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

    # 1. 전방 깊이 축 투영 거리 및 매끄러운 돔 감쇠
    t_proj = pts_rel @ bone_u
    t_norm = np.clip(t_proj / bone_len, 0.0, 1.0)
    w_depth = 0.25 + 0.75 * np.sin(np.sqrt(t_norm) * (np.pi * 0.5))

    # 2. 본 세그먼트 [Head -> Tail] 와의 3D 수직 최단거리 및 Hann Cosine 감쇠
    proj_pts = head_arr + np.outer(t_norm, bone_vec)
    d_perp = np.linalg.norm(world_pts - proj_pts, axis=1)

    radius = max(0.02, float(eff_radius))
    rho_norm = np.clip(d_perp / radius, 0.0, 1.0)
    w_radial = np.where(d_perp < radius, 0.5 * (1.0 + np.cos(np.pi * rho_norm)), 0.0)

    # 3. 해부학적 합성 가중치 계산 (반경 밖은 정확히 0.0)
    w_comb = peak_w * np.power(np.maximum(0.0, w_depth * w_radial), falloff_p)
    w_arr = np.where(w_radial > 0.0, np.clip(w_comb, 0.001, 1.0), 0.0)

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
    w_depth = 0.25 + 0.75 * math.sin(math.sqrt(t_norm) * (math.pi * 0.5))

    if props:
        b_params = get_bone_params(props, bname)
        peak_w = b_params["peak_weight"]
        falloff_p = b_params["falloff_power"]

    w = peak_w * math.pow(max(0.0, w_depth * w_radial), falloff_p)
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


def smooth_mesh_weights(obj, vg_list, rigid_mask, iters=2, alpha=0.45, lock_registered=True):
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


def synchronize_coincident_vertices_across_objects(target_objs, vg_names=None, tol=0.0012):
    """
    모든 대상 오브젝트에 걸쳐 3D 공간 상에서 동일한 위치(기본 1.2mm 이내)를 공유하는
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


def rebake_mesh_weights_from_rig(context, rig, props=None):
    """
    물리 리그(XXMI_Physics_Rig) 내의 본 위치를 기준으로
    씬 내의 모든 관련 메쉬 오브젝트들의 버텍스 가중치를 동적으로 다시 계산합니다.
    사용자가 등록한 개별 모핑 본 슬롯 및 본 위치 수정을 100% 반영합니다.
    """
    if props is None:
        props = context.scene.xxmi_custom_physics_props

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

            if side_lock_registered:
                cand_mask = is_registered_b & ~rigid_mask
            else:
                dist = np.linalg.norm(world - cent, axis=1)
                in_radius = (dist <= eff_radius)
                dot_fwd = (world - head) @ fwd_u
                front_ok = (dot_fwd >= -eff_radius * 0.35)
                cand_mask = (is_registered_b | (in_radius & front_ok)) & ~rigid_mask

                non_cand_prev = np.nonzero(is_registered_b & ~cand_mask)[0]
                if len(non_cand_prev) > 0:
                    zero_entries.append((vg, non_cand_prev.tolist()))

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

        if not side_lock_registered:
            for vg, z_list in zero_entries:
                vg.remove(z_list)

    # 4. 라플라시안 스무딩 (모든 활성 모핑 본 버텍스 그룹 대상)
    for bname in active_bones:
        b_params = get_bone_params(props, bname)
        s_iters = b_params["smooth_iterations"]
        if s_iters > 0:
            vg_name = f"XXMI_{bname}"
            for obj, odata in obj_data.items():
                vg = obj.vertex_groups.get(vg_name)
                if vg:
                    smooth_mesh_weights(
                        obj, [vg], odata["rigid_mask"],
                        iters=s_iters, alpha=0.45, lock_registered=lock_registered
                    )

    # 5. 동일 위치 공유 정점 동기화
    num_synced = synchronize_coincident_vertices_across_objects(
        target_objs, vg_names=[f"XXMI_{bn}" for bn in active_bones], tol=0.0012
    )

    # 6. 강체(액세서리) 표면 밀착 가중치 동기화
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

    # 다중 강체 슬롯 가중치 계산 및 할당
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
                        slot_pts.append(Vector(r_world[vi]))

        c_cent = Vector(np.mean([p[:] for p in slot_pts], axis=0)) if slot_pts else Vector((0.0, 0.0, 0.0))

        # 목표 본 결정
        t_bone = item.target_bone
        best_bname = None
        if t_bone != 'AUTO' and t_bone in active_bones:
            best_bname = t_bone
        elif t_bone != 'AUTO' and (f"XXMI_{t_bone}" in [f"XXMI_{b}" for b in active_bones]):
            best_bname = t_bone
        else:
            # AUTO: 강체 중심과 가장 가까운 모핑 본 선택
            min_dist = float('inf')
            for bn in active_bones:
                bentry = bone_info.get(bn)
                if bentry:
                    b_cent = Vector(bentry["centroid"])
                    d = (c_cent - b_cent).length
                    if d < min_dist:
                        min_dist = d
                        best_bname = bn

        if not best_bname and active_bones:
            best_bname = active_bones[0]

        if not best_bname:
            continue

        # 표면 가중치 탐색
        surf_w = 0.0
        if best_bname in bone_kd_map and slot_pts:
            kd, s_weights = bone_kd_map[best_bname]
            hits = kd.find_range(c_cent, 0.08)
            if not hits:
                hits = [kd.find(c_cent)]
            valid_hits = [h for h in hits if h is not None and h[0] is not None]
            if valid_hits:
                ws = [s_weights[h[1]] for h in valid_hits if s_weights[h[1]] > 0.001]
                if ws:
                    surf_w = float(np.mean(ws))

        anat_w = eval_bone_weight_at_point(c_cent, best_bname, bone_info, props)
        base_w = surf_w if surf_w > 0.01 else anat_w
        w_scale = getattr(item, "weight_scale", 1.0)
        final_rigid_w = min(1.0, max(0.02, base_w * w_scale))

        target_vg_name = f"XXMI_{best_bname}"

        for oname, vis in rdata.items():
            robj = bpy.data.objects.get(oname)
            if not robj or robj.type != 'MESH' or not vis:
                continue

            vg_target = robj.vertex_groups.get(target_vg_name) or robj.vertex_groups.new(name=target_vg_name)
            vg_rigid = robj.vertex_groups.get(VG_RIGID) or robj.vertex_groups.new(name=VG_RIGID)

            vg_rigid.add(vis, 1.0, 'REPLACE')
            vg_target.add(vis, final_rigid_w, 'REPLACE')

            # 다른 모핑 본 그룹에서는 제외하여 강체가 왜곡되지 않도록 보장
            for obn in active_bones:
                if obn != best_bname:
                    o_vg = robj.vertex_groups.get(f"XXMI_{obn}")
                    if o_vg:
                        o_vg.remove(vis)

        applied_slots += 1

    return updated_obj_count, f"{updated_obj_count}개 메쉬 ({total_recomputed_verts}개 정점, 동기화 {num_synced}개, 강체슬롯 {applied_slots}개) 가중치 갱신 완료"


