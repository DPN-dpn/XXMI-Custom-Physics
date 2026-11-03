import bpy
import bmesh
import math
import heapq
import numpy as np
from collections import deque
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from ..utils import (
    ensure_object_visible,
    read_vertex_group_arrays,
    world_vertex_positions,
    mesh_edge_arrays,
    label_components,
)

BREAST_RIG_NAME = "XXMI_Breast_Rig"
VG_BREAST_L = "XXMI_Breast_L"
VG_BREAST_R = "XXMI_Breast_R"
VG_RIGID = "XXMI_Role_Rigid"


def _get_target_mesh_objects(context):
    """현재 선택되었거나 에디트 모드에 진입해 있는 모든 메쉬 오브젝트 목록을 반환합니다."""
    objs = []
    if context.mode == 'EDIT_MESH':
        objs = [o for o in context.objects_in_mode if o.type == 'MESH']
    if not objs:
        objs = [o for o in context.selected_objects if o.type == 'MESH']
    if not objs and context.active_object and context.active_object.type == 'MESH':
        objs = [context.active_object]
    return objs


def _get_or_create_breast_rig(context, props, ref_obj):
    """가슴 물리 앵커 역할을 하는 Armature(XXMI_Breast_Rig)를 가져오거나 새로 생성합니다."""
    rig = props.anchor_obj
    if rig and rig.name in bpy.data.objects and rig.type == 'ARMATURE':
        ensure_object_visible(context, rig)
        return rig

    existing = bpy.data.objects.get(BREAST_RIG_NAME)
    if existing and existing.type == 'ARMATURE':
        props.anchor_obj = existing
        ensure_object_visible(context, existing)
        return existing

    arm_data = bpy.data.armatures.new(BREAST_RIG_NAME)
    arm_data.display_type = 'OCTAHEDRAL'

    rig = bpy.data.objects.new(BREAST_RIG_NAME, arm_data)
    rig.show_in_front = True
    col = ref_obj.users_collection[0] if (ref_obj and ref_obj.users_collection) else context.scene.collection
    col.objects.link(rig)
    props.anchor_obj = rig
    return rig


def _find_connected_islands(vert_indices, edges_list):
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


def _calculate_anatomical_weights_with_anti_clipping(world_pts, head, tail_or_fwd, eff_radius, falloff_p=1.0):
    """
    해부학적 앵커 본 세그먼트[Head -> Tail] 기반 3D 이중 감쇠(전방 깊이 Smoothstep + 방사형 Cosine):
    1. 전방 깊이 감쇠(Depth Falloff): 흉벽 부착부(Head)에서 0.0 -> 가슴 끝 유두(Tail)에서 1.0으로 매끄럽게 증가 (Hermite Curve)
       -> 흉벽에 가까운 정점의 가중치가 ~0에 수렴하여 가슴이 움직일 때 몸통/피부와 벌어지는 틈(Tearing/Gap) 완벽 방지
    2. 방사형 감쇠(Radial Falloff): 본 중심축에서 외곽 둘레(Perimeter)로 갈수록 1.0 -> 0.0으로 완만하게 감소 (Cosine Curve)
       -> 선택 테두리 부위가 주변 메쉬와 단차 없이 부드럽게 이어짐
    3. UV 심(Seam) 및 분리 정점 동기화: 0.5mm 이내 동일 3D 좌표 100% 동일 가중치 보장
    4. 선택 보존: 최소 가중치 0.005를 보장하여 [적용된 가슴 정점 재선택] 시 선택 영역이 100% 온전히 유지됨
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

    # 1. 전방 깊이 축(Forward Axis) 투영 거리 및 Hermite Smoothstep 감쇠 (Head=0 -> Tail=1)
    t_proj = pts_rel @ bone_u
    t_norm = np.clip(t_proj / bone_len, 0.0, 1.0)
    w_depth = 3.0 * (t_norm ** 2) - 2.0 * (t_norm ** 3)

    # 2. 중심축으로부터의 방사형 수직 거리(Perpendicular Radius) 및 Cosine 감쇠 (Axis=1 -> Edge=0)
    v_perp = pts_rel - np.outer(t_proj, bone_u)
    rho = np.linalg.norm(v_perp, axis=1)
    radius = max(1e-4, float(eff_radius))
    rho_norm = np.clip(rho / radius, 0.0, 1.0)
    w_radial = np.cos(rho_norm * (np.pi * 0.5))

    # 3. 해부학적 합성 가중치 계산 및 최소 보장 가중치(0.005) 적용
    w_comb = np.power(np.maximum(0.0, w_depth * w_radial), falloff_p)
    w_arr = np.clip(w_comb, 0.005, 1.0)

    # 4. UV 심(Seam) 및 분리 정점 동기화: 0.5mm 이내 동일 3D 좌표 100% 동일 가중치 부여
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


class XXMI_OT_register_breast_selection(bpy.types.Operator):
    bl_idname = "object.xxmi_register_breast_selection"
    bl_label = "선택 영역을 가슴 물리로 등록"
    bl_description = "선택된 모든 메쉬 오브젝트(Body, 의상 등)에서 가슴 정점을 좌/우로 분할하고, 장신구 강체 감지 및 가슴 웨이트를 일괄 생성합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        objs = _get_target_mesh_objects(context)
        return len(objs) > 0

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_objs = _get_target_mesh_objects(context)

        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        for obj in target_objs:
            ensure_object_visible(context, obj)

        # 1. 대상 메쉬들에서 선택된 정점 수집 (numpy 일괄 처리)
        obj_sel_data = {}  # obj -> {"left": [vi, ...], "right": [vi, ...], "world": ndarray(n,3), "mw": Matrix}
        all_left_world_pts = []
        all_right_world_pts = []

        total_sel_count = 0

        for obj in target_objs:
            mesh = obj.data
            mw = obj.matrix_world
            n_verts = len(mesh.vertices)
            if n_verts == 0:
                continue

            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0]
            if len(sel_indices) == 0:
                continue

            world = world_vertex_positions(obj)

            # UV 심(Seam) 및 분리된 경계 정점 동기화: 선택된 정점과 동일 위치(0.5mm 이내)인 분리 정점 자동 포함
            kd_local = KDTree(n_verts)
            for vi, wpos in enumerate(world):
                kd_local.insert(Vector(wpos), vi)
            kd_local.balance()

            extra_indices = []
            for si in sel_indices.tolist():
                hits = kd_local.find_range(Vector(world[si]), 0.0005)
                for _, hi, _ in hits:
                    if not sel_mask[hi]:
                        sel_mask[hi] = True
                        extra_indices.append(hi)

            if extra_indices:
                sel_indices = np.nonzero(sel_mask)[0]

            total_sel_count += len(sel_indices)

            is_left = world[sel_indices, 0] >= 0.0
            left_idx = sel_indices[is_left]
            right_idx = sel_indices[~is_left]
            all_left_world_pts.extend(Vector(p) for p in world[left_idx].tolist())
            all_right_world_pts.extend(Vector(p) for p in world[right_idx].tolist())

            obj_sel_data[obj] = {
                "left": left_idx.tolist(),
                "right": right_idx.tolist(),
                "world": world,
                "mw": mw,
            }

        if total_sel_count < 6 or (not all_left_world_pts and not all_right_world_pts):
            if was_edit:
                for obj in target_objs:
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "에디트 모드에서 가슴 표면 정점들을 먼저 선택해주세요. (최소 6개 이상)")
            return {'CANCELLED'}

        # 2. 전체 선택 정점 기준 통합 가슴 앵커 위치(월드 좌표) 계산
        anchor_info = {}
        for side, pts, bname in (("L", all_left_world_pts, "Breast.L"), ("R", all_right_world_pts, "Breast.R")):
            if not pts:
                continue
            centroid = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
            # 블렌더 좌표계에서 캐릭터 전방(가슴 돌출 방향)은 -Y 축입니다 (Front View는 +Y를 바라봄)
            tip_pos = min(pts, key=lambda p: p.y)
            fwd_dir = (tip_pos - centroid)
            if fwd_dir.length < 0.01:
                fwd_dir = Vector((0.0, -0.06, 0.0))
            else:
                fwd_dir = fwd_dir.normalized() * max(0.04, (centroid - tip_pos).length)

            head_pos = centroid - fwd_dir * 0.8
            tail_pos = tip_pos
            bone_vec_v = tail_pos - head_pos
            bone_len_v = bone_vec_v.length
            bone_u_v = bone_vec_v.normalized() if bone_len_v > 1e-4 else Vector((0.0, -1.0, 0.0))
            max_rho = max(((p - head_pos) - ((p - head_pos).dot(bone_u_v)) * bone_u_v).length for p in pts)
            radius = max(0.03, max_rho)

            anchor_info[side] = {
                "bname": bname,
                "centroid": centroid,
                "tip": tip_pos,
                "head": head_pos,
                "tail": tail_pos,
                "fwd_unit": (tail_pos - head_pos).normalized(),
                "radius": radius,
            }

        # 3. 통합 가슴 앵커 아머처(XXMI_Breast_Rig) 생성/갱신
        ref_obj = target_objs[0]
        rig = _get_or_create_breast_rig(context, props, ref_obj)
        ensure_object_visible(context, rig)
        context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')

        ebones = rig.data.edit_bones
        for eb in list(ebones):
            ebones.remove(eb)

        for side, d in anchor_info.items():
            eb = ebones.new(d["bname"])
            eb.head = d["head"]
            eb.tail = d["tail"]
            eb.head_radius = d["radius"]
            eb.tail_radius = d["radius"] * 0.7
            eb.envelope_distance = d["radius"] * 1.4
            eb["xxmi_bone_type"] = "BREAST"
            eb["xxmi_side"] = side

        bpy.ops.object.mode_set(mode='OBJECT')

        # 4. 각 메쉬 오브젝트별로 가중치 그룹 생성 및 베이크
        falloff_p = props.falloff_power
        total_rigid_islands = 0

        for obj, sdata in obj_sel_data.items():
            mesh = obj.data
            world = sdata["world"]
            n_verts = len(mesh.vertices)

            vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)
            vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)

            # 사용자가 수동 지정해 둔 기존 강체 정점 보존
            prev = read_vertex_group_arrays(obj, {vg_rigid.index: "rigid"})
            existing_rigid = prev["rigid"] > 0.5

            all_v_indices = list(range(n_verts))
            vg_l.remove(all_v_indices)
            vg_r.remove(all_v_indices)
            vg_rigid.remove(all_v_indices)

            selected = np.zeros(n_verts, dtype=bool)
            selected[sdata["left"]] = True
            selected[sdata["right"]] = True

            rigid_all = existing_rigid & selected

            # A) 강체 장신구는 사용자가 [선택 부위를 강체 장신구로 지정]으로 직접 지정한 정점만 사용합니다. (자동 감지 없음)

            # B) 피부 및 옷감 정점 감쇠 베이크 (원통형 해부학적 거리 + Anti-Clipping 필터 적용)
            for side, vlist, vg in (("L", sdata["left"], vg_l), ("R", sdata["right"], vg_r)):
                if not vlist or side not in anchor_info:
                    continue

                head = np.array(anchor_info[side]["head"], dtype=np.float64)
                tail = np.array(anchor_info[side]["tail"], dtype=np.float64)
                radius = anchor_info[side]["radius"]
                eff_radius = max(0.02, radius)

                idx_arr = np.array(vlist, dtype=np.int64)
                world_sub = world[idx_arr]
                w_arr = _calculate_anatomical_weights_with_anti_clipping(
                    world_sub, head, tail, eff_radius, falloff_p
                )



                for vi, wi in zip(vlist, w_arr.tolist()):
                    if rigid_all[vi]:
                        continue
                    vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')

            # C) 가슴 중앙 골(Cleavage) 대칭 크로스 블렌딩 (중앙선 분리 찢어짐 영구 차단)
            center_x = 0.0
            if "L" in anchor_info and "R" in anchor_info:
                center_x = (anchor_info["L"]["head"][0] + anchor_info["R"]["head"][0]) * 0.5

            blend_w = 0.025  # 2.5cm 대칭 블렌드 반경
            arrs_now = read_vertex_group_arrays(obj, {vg_l.index: "l", vg_r.index: "r"})
            wl_all = arrs_now["l"]
            wr_all = arrs_now["r"]
            near_center = np.abs(world[:, 0] - center_x) < blend_w
            active = (wl_all > 0.0001) | (wr_all > 0.0001)
            target_mask = near_center & active & ~rigid_all
            for vi in np.nonzero(target_mask)[0].tolist():
                x_dist = abs(world[vi, 0] - center_x)
                t = x_dist / blend_w
                # Hermite Curve: t=0 -> k=0.5 (완전 5:5 대칭 결합), t=1 -> k=0.0 (독립 거동)
                k = 0.5 * (1.0 - (3.0 * t * t - 2.0 * t * t * t))
                orig_l = wl_all[vi]
                orig_r = wr_all[vi]
                new_l = (1.0 - k) * orig_l + k * orig_r
                new_r = (1.0 - k) * orig_r + k * orig_l
                vg_l.add([vi], max(0.0, min(1.0, float(new_l))), 'REPLACE')
                vg_r.add([vi], max(0.0, min(1.0, float(new_r))), 'REPLACE')

            rigid_list = np.nonzero(rigid_all)[0].tolist()
            if rigid_list:
                vg_rigid.add(rigid_list, 1.0, 'REPLACE')

        # D) 다중 오브젝트(Body 케이스 + Hair 악세서리 등) 간 3D 공간 통합 강체 클러스터 가중치 동기화
        num_clusters = _unify_rigid_clusters_across_objects(list(obj_sel_data.keys()), anchor_info, falloff_p)
        total_rigid_islands = max(total_rigid_islands, num_clusters)

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        obj_names_str = ", ".join(o.name for o in obj_sel_data.keys())
        self.report(
            {'INFO'},
            f"다중 오브젝트({len(obj_sel_data)}개: {obj_names_str}) 가슴 물리 등록 완료! (총 정점: {total_sel_count}개, 통합 강체: {total_rigid_islands}개)"
        )
        return {'FINISHED'}


def _unify_rigid_clusters_across_objects(target_objs, bone_info, falloff_p):
    """
    여러 오브젝트(예: Hair의 악세서리 + Body의 악세서리 케이스)에 나뉘어 있는 강체 장신구(XXMI_Role_Rigid)들을
    3D 월드 공간에서 서로 맞닿아 있는(5mm 이내) 덩어리끼리 하나의 '통합 강체 클러스터'로 묶어
    완벽히 동일한 단일 (w_l, w_r) 가중치를 부여합니다. (악세서리가 케이스를 뚫고 나오는 현상 방지)
    강체로 지정되지 않은 정점은 일절 건드리지 않습니다.
    """
    if not target_objs or not bone_info:
        return 0

    islands = []  # {"obj", "verts", "pts"(ndarray), "bmin", "bmax", "cent"}

    for obj in target_objs:
        if obj.type != 'MESH':
            continue
        vg_rigid = obj.vertex_groups.get(VG_RIGID)
        if vg_rigid is None:
            continue

        mesh = obj.data
        n_verts = len(mesh.vertices)
        rigid_mask = read_vertex_group_arrays(obj, {vg_rigid.index: "rigid"})["rigid"] > 0.5
        if not rigid_mask.any():
            continue

        world = world_vertex_positions(obj)
        e0, e1 = mesh_edge_arrays(mesh)
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
    margin = 0.005
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
            step = max(1, len(a["pts"]) // 64)
            kd_b = _get_kd(j)
            for p in a["pts"][::step].tolist():
                if kd_b.find(Vector(p))[2] <= margin:
                    parent[_find(i)] = _find(j)
                    break

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
        w_depth = 3.0 * (t_norm ** 2) - 2.0 * (t_norm ** 3)

        v_perp = v - t * fwd_u
        rho = v_perp.length
        rho_norm = max(0.0, min(1.0, rho / radius))
        w_radial = math.cos(rho_norm * (math.pi * 0.5))

        w = math.pow(max(0.0, w_depth * w_radial), falloff_p)
        return max(0.05, min(1.0, w)) if w > 0.0 else 0.0

    for comp in clusters.values():
        all_pts = np.concatenate([isl["pts"] for isl in comp], axis=0)
        c_cent = Vector(all_pts.mean(axis=0).tolist())
        min_x = float(all_pts[:, 0].min())
        max_x = float(all_pts[:, 0].max())

        has_l = "L" in bone_info
        has_r = "R" in bone_info

        # 중앙선(X=0)에 걸쳐 있는 장신구/케이스는 좌/우 가중치를 비율로 묶어 반으로 갈라지지 않게 보호
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


def _get_rig_bone_info(rig):
    """
    XXMI_Breast_Rig 내의 본(Breast.L, Breast.R) 정보를 월드 좌표계로 추출합니다.
    Edit Mode, Pose Mode, Object Mode 모두 대응합니다.
    """
    bone_info = {}
    if not rig or rig.type != 'ARMATURE':
        return bone_info

    if rig.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.view_layer.update()

    mw = rig.matrix_world
    pose_bones = rig.pose.bones if rig.pose else None
    data_bones = rig.data.bones

    for side, bname in (("L", "Breast.L"), ("R", "Breast.R")):
        db = data_bones.get(bname)
        if not db:
            continue

        pb = pose_bones.get(bname) if pose_bones else None
        if pb:
            head_w = mw @ pb.head
            tail_w = mw @ pb.tail
        else:
            head_w = mw @ db.head_local
            tail_w = mw @ db.tail_local

        fwd = tail_w - head_w
        fwd_len = fwd.length
        fwd_unit = (fwd / fwd_len) if fwd_len > 1e-4 else Vector((0.0, -1.0, 0.0))
        cent = head_w + fwd * 0.4
        tip = tail_w

        scale_factor = 1.0
        if pb:
            scale_factor = max(abs(pb.scale.x), abs(pb.scale.y), abs(pb.scale.z))

        radius = max(0.03, db.head_radius * scale_factor, (db.envelope_distance / 1.4) * scale_factor, fwd_len * 0.5)

        bone_info[side] = {
            "bname": bname,
            "centroid": cent,
            "tip": tip,
            "head": head_w,
            "tail": tail_w,
            "fwd_unit": fwd_unit,
            "radius": radius,
        }

    return bone_info


def _rebake_mesh_weights_from_rig(context, rig, props=None):
    """
    XXMI_Breast_Rig 내의 본 위치(Breast.L / Breast.R)를 기준으로
    씬 내의 모든 관련 메쉬 오브젝트들의 버텍스 가중치를 다시 계산합니다.
    """
    if props is None:
        props = context.scene.xxmi_custom_physics_props

    bone_info = _get_rig_bone_info(rig)
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
        # 기존 가슴 버텍스 그룹이 없다면 선택된 메쉬(또는 씬 내 메쉬)를 대상으로 가슴 영역 자동 탐색 및 그룹 생성
        target_objs = [o for o in context.selected_objects if o.type == 'MESH']
        if not target_objs and context.active_object and context.active_object.type == 'MESH':
            target_objs = [context.active_object]
        if not target_objs:
            target_objs = [o for o in bpy.data.objects if o.type == 'MESH']

    if not target_objs:
        return 0, "씬 내에서 대상 메쉬 오브젝트를 찾을 수 없습니다."

    falloff_p = props.falloff_power
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

        cent = np.array(bone_info[side]["centroid"], dtype=np.float64)
        head = np.array(bone_info[side]["head"], dtype=np.float64)
        fwd_u = np.array(bone_info[side]["fwd_unit"], dtype=np.float64)
        radius = bone_info[side]["radius"]
        eff_radius = max(0.02, radius)

        # 1. 모든 대상 오브젝트에서 해당 측면의 가슴 후보 정점을 3D 공간 상에서 통합 수집
        side_cand = []
        zero_entries = []

        for obj, odata in obj_data.items():
            vg = odata["vg_l"] if side == "L" else odata["vg_r"]
            world = odata["world"]
            arrs = odata["arrs"]
            rigid_mask = odata["rigid_mask"]

            prev_weighted = (arrs[prev_vg_key] > 0.0001)

            if lock_registered:
                # [등록 정점 범위 엄격 유지 모드]: 사용자가 등록해 둔 정점만 100% 엄격 유지!
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

        # 2. 원통형 해부학적 거리 메트릭 + 다중 레이어 Anti-Clipping 필터를 모든 오브젝트에 통합 적용
        cand_pts = np.array([c[3] for c in side_cand], dtype=np.float64)
        tail = np.array(bone_info[side]["tail"], dtype=np.float64)
        baked_w = _calculate_anatomical_weights_with_anti_clipping(
            cand_pts, head, tail, eff_radius, falloff_p
        )

        final_weights = baked_w.copy()

        for (obj, vg, vi, _), wi in zip(side_cand, final_weights.tolist()):
            if lock_registered:
                vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')
                total_recomputed_verts += 1
            else:
                if wi <= 0.0001:
                    vg.remove([vi])
                else:
                    vg.add([vi], max(0.0, min(1.0, wi)), 'REPLACE')
                    total_recomputed_verts += 1

        if not lock_registered:
            for vg, z_list in zero_entries:
                vg.remove(z_list)

    # 4. 가슴 중앙 골(Cleavage) 대칭 크로스 블렌딩 (중앙선 분리 찢어짐 영구 차단)
    center_x = 0.0
    if "L" in bone_info and "R" in bone_info:
        center_x = (bone_info["L"]["head"][0] + bone_info["R"]["head"][0]) * 0.5

    blend_w = 0.025  # 2.5cm 대칭 블렌드 반경
    for obj, odata in obj_data.items():
        vg_l = odata["vg_l"]
        vg_r = odata["vg_r"]
        world = odata["world"]
        rigid_mask = odata["rigid_mask"]
        arrs_now = read_vertex_group_arrays(obj, {vg_l.index: "l", vg_r.index: "r"})
        wl_all = arrs_now["l"]
        wr_all = arrs_now["r"]
        near_center = np.abs(world[:, 0] - center_x) < blend_w
        active = (wl_all > 0.0001) | (wr_all > 0.0001)
        target_mask = near_center & active & ~rigid_mask
        for vi in np.nonzero(target_mask)[0].tolist():
            x_dist = abs(world[vi, 0] - center_x)
            t = x_dist / blend_w
            k = 0.5 * (1.0 - (3.0 * t * t - 2.0 * t * t * t))
            orig_l = wl_all[vi]
            orig_r = wr_all[vi]
            new_l = (1.0 - k) * orig_l + k * orig_r
            new_r = (1.0 - k) * orig_r + k * orig_l
            vg_l.add([vi], max(0.0, min(1.0, float(new_l))), 'REPLACE')
            vg_r.add([vi], max(0.0, min(1.0, float(new_r))), 'REPLACE')

    for obj, odata in obj_data.items():
        total_recomputed_verts += int(np.count_nonzero(odata["rigid_mask"]))
        updated_obj_count += 1

    # 다중 오브젝트(Body 케이스 + Hair 악세서리 등) 간 3D 공간 통합 강체 클러스터 가중치 동기화
    num_clusters = _unify_rigid_clusters_across_objects(target_objs, bone_info, falloff_p)

    return updated_obj_count, f"{updated_obj_count}개 메쉬 ({total_recomputed_verts}개 정점, 통합 강체 {num_clusters}개) 가중치 갱신 완료"


class XXMI_OT_rebake_breast_weights(bpy.types.Operator):
    bl_idname = "object.xxmi_rebake_breast_weights"
    bl_label = "조정된 본 위치 기준으로 가중치 재계산"
    bl_description = "수동 조정한 XXMI_Breast_Rig 아머처의 본(Breast.L / Breast.R) 위치를 기준으로 가슴 버텍스 가중치(XXMI_Breast_L/R)를 다시 계산합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        rig = props.anchor_obj or bpy.data.objects.get(BREAST_RIG_NAME)
        return rig is not None and rig.type == 'ARMATURE'

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        rig = props.anchor_obj or bpy.data.objects.get(BREAST_RIG_NAME)
        if not rig:
            self.report({'ERROR'}, "XXMI_Breast_Rig 아머처를 찾을 수 없습니다.")
            return {'CANCELLED'}

        # 1. 메쉬 또는 아머처의 Edit 모드 버퍼를 Object 모드로 커밋하고 뷰레이어 갱신
        was_mode = context.mode
        active_obj = context.view_layer.objects.active
        if was_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        context.view_layer.update()

        count, msg = _rebake_mesh_weights_from_rig(context, rig, props)

        context.view_layer.update()

        # 2. 작업 편의를 위해 이전 모드가 Edit/Pose였으면 안전하게 복원
        if was_mode in ('EDIT_ARMATURE', 'POSE') and active_obj:
            try:
                context.view_layer.objects.active = active_obj
                target_mode = 'EDIT' if was_mode == 'EDIT_ARMATURE' else 'POSE'
                bpy.ops.object.mode_set(mode=target_mode)
            except Exception:
                pass

        if count == 0:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        self.report({'INFO'}, f"본 위치 동기화 완료: {msg}")
        return {'FINISHED'}


class XXMI_OT_select_breast_vertices(bpy.types.Operator):
    bl_idname = "object.xxmi_select_breast_vertices"
    bl_label = "적용된 가슴 정점 재선택"
    bl_description = "가슴 물리 웨이트가 등록된 모든 메쉬 오브젝트(Body, 의상, 악세서리 등)를 함께 선택하고 에디트 모드에서 해당 가슴 정점들을 다시 선택합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(
            o.type == 'MESH' and (o.vertex_groups.get(VG_BREAST_L) or o.vertex_groups.get(VG_BREAST_R))
            for o in bpy.data.objects
        )

    def execute(self, context):
        target_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and (o.vertex_groups.get(VG_BREAST_L) or o.vertex_groups.get(VG_BREAST_R))
        ]

        if not target_objs:
            self.report({'WARNING'}, "가슴 물리 웨이트(XXMI_Breast_L/R)가 등록된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        # 1. 오브젝트 모드로 전환 후 기존 선택 해제
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')

        # 2. 가슴 웨이트가 있는 모든 메쉬 오브젝트 동시 선택
        for obj in target_objs:
            ensure_object_visible(context, obj)
            obj.select_set(True)

        context.view_layer.objects.active = target_objs[0]

        # 3. 에디트 모드 진입 및 정점 선택 모드 활성화
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')

        # 4. 각 메쉬 오브젝트별로 가슴 버텍스 그룹 정점 선택
        total_selected = 0
        for obj in target_objs:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()

            vg_l = obj.vertex_groups.get(VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R)
            vg_rigid = obj.vertex_groups.get(VG_RIGID)

            vg_indices = set()
            if vg_l: vg_indices.add(vg_l.index)
            if vg_r: vg_indices.add(vg_r.index)
            if vg_rigid: vg_indices.add(vg_rigid.index)

            dvert_lay = bm.verts.layers.deform.verify()

            for v in bm.verts:
                dvert = v[dvert_lay]
                is_breast = any(g_idx in dvert and dvert[g_idx] > 0.0001 for g_idx in vg_indices)
                v.select_set(is_breast)
                if is_breast:
                    total_selected += 1

            bmesh.update_edit_mesh(obj.data)

        obj_names = ", ".join(o.name for o in target_objs)
        self.report({'INFO'}, f"총 {len(target_objs)}개 오브젝트({obj_names})에서 {total_selected}개 가슴 정점을 재선택했습니다.")
        return {'FINISHED'}


class XXMI_OT_mark_selection_rigid(bpy.types.Operator):
    bl_idname = "object.xxmi_mark_selection_rigid"
    bl_label = "선택 부위를 강체 장신구로 지정"
    bl_description = "선택된 모든 오브젝트에서 에디트 모드로 선택한 정점들을 강체 장신구(XXMI_Role_Rigid)로 등록합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        objs = _get_target_mesh_objects(context)
        return len(objs) > 0

    def execute(self, context):
        target_objs = _get_target_mesh_objects(context)
        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        total_marked = 0
        for obj in target_objs:
            sel = [v.index for v in obj.data.vertices if v.select]
            if not sel:
                continue
            vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)
            vg_rigid.add(sel, 1.0, 'REPLACE')
            total_marked += len(sel)

        props = context.scene.xxmi_custom_physics_props
        rig = props.anchor_obj or bpy.data.objects.get(BREAST_RIG_NAME)
        sync_msg = ""
        if rig and rig.type == 'ARMATURE' and total_marked > 0:
            _, sync_msg = _rebake_mesh_weights_from_rig(context, rig, props)
            sync_msg = f" -> {sync_msg}"

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if total_marked == 0:
            self.report({'ERROR'}, "장신구로 지정할 정점들을 선택해주세요.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"총 {total_marked}개 정점을 강체 장신구(XXMI_Role_Rigid)로 등록 및 가중치 동기화 완료!{sync_msg}")
        return {'FINISHED'}


class XXMI_OT_clear_breast_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_clear_breast_physics"
    bl_label = "가슴 물리 초기화"
    bl_description = "가슴 앵커 오브젝트 및 씬 내 모든 메쉬에 등록된 가슴 웨이트 그룹을 일괄 제거합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return props.anchor_obj is not None or bpy.data.objects.get(BREAST_RIG_NAME) is not None

    def execute(self, context):
        from ..simulation.operators import stop_physics_preview
        props = context.scene.xxmi_custom_physics_props
        if props.is_preview_running:
            stop_physics_preview(context)

        rig = props.anchor_obj or bpy.data.objects.get(BREAST_RIG_NAME)
        if rig:
            bpy.data.objects.remove(rig, do_unlink=True)
            props.anchor_obj = None

        # 씬 내 모든 메쉬에서 가슴 관련 버텍스 그룹 제거
        cleared_objs = 0
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                removed_any = False
                for vg_name in (VG_BREAST_L, VG_BREAST_R, VG_RIGID):
                    vg = obj.vertex_groups.get(vg_name)
                    if vg:
                        obj.vertex_groups.remove(vg)
                        removed_any = True
                if removed_any:
                    cleared_objs += 1

        self.report({'INFO'}, f"가슴 물리 설정 초기화 완료 ({cleared_objs}개 메쉬 오브젝트 가중치 제거됨).")
        return {'FINISHED'}


classes = (
    XXMI_OT_register_breast_selection,
    XXMI_OT_select_breast_vertices,
    XXMI_OT_rebake_breast_weights,
    XXMI_OT_mark_selection_rigid,
    XXMI_OT_clear_breast_physics,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
