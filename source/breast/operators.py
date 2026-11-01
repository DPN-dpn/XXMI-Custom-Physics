import bpy
import bmesh
import math
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from ..utils import ensure_object_visible

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
            queue = [v]
            visited.add(v)
            while queue:
                curr = queue.pop(0)
                island.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            islands.append(island)

    return islands


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

        # 1. 대상 메쉬들에서 선택된 정점 수집
        obj_sel_data = {}  # obj -> {"left": [vi, ...], "right": [vi, ...], "edges": [...], "mw": Matrix}
        all_left_world_pts = []
        all_right_world_pts = []

        total_sel_count = 0

        for obj in target_objs:
            mesh = obj.data
            mw = obj.matrix_world
            sel_indices = [v.index for v in mesh.vertices if v.select]
            if not sel_indices:
                continue

            total_sel_count += len(sel_indices)
            edges_list = [(e.vertices[0], e.vertices[1]) for e in mesh.edges]

            left_verts = []
            right_verts = []
            for vi in sel_indices:
                w_pos = mw @ mesh.vertices[vi].co
                if w_pos.x >= 0.0:
                    left_verts.append(vi)
                    all_left_world_pts.append(w_pos)
                else:
                    right_verts.append(vi)
                    all_right_world_pts.append(w_pos)

            obj_sel_data[obj] = {
                "left": left_verts,
                "right": right_verts,
                "edges": edges_list,
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
            # 표면 전방(+Y) 방향으로 가장 튀어나온 꼭지점 탐색
            tip_pos = max(pts, key=lambda p: p.y)
            fwd_dir = (tip_pos - centroid)
            if fwd_dir.length < 0.01:
                fwd_dir = Vector((0.0, 0.08, 0.0))
            else:
                fwd_dir = fwd_dir.normalized() * max(0.04, (tip_pos - centroid).length * 1.2)

            head_pos = centroid - fwd_dir * 0.4
            tail_pos = centroid + fwd_dir * 0.6
            radius = max(0.03, (centroid - tip_pos).length * 0.75)

            anchor_info[side] = {
                "bname": bname,
                "centroid": centroid,
                "tip": tip_pos,
                "head": head_pos,
                "tail": tail_pos,
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

        # 다중 오브젝트(Body, Hair 등) 전체에서 가장 큰 메쉬 섬(메인 가슴 표면) 크기 산출
        global_max_island_len = {"L": 0, "R": 0}
        obj_side_islands = {}
        for obj, sdata in obj_sel_data.items():
            edges_list = sdata["edges"]
            obj_side_islands[obj] = {}
            for side, vlist in (("L", sdata["left"]), ("R", sdata["right"])):
                if not vlist:
                    obj_side_islands[obj][side] = []
                    continue
                isls = _find_connected_islands(vlist, edges_list)
                isls.sort(key=len, reverse=True)
                obj_side_islands[obj][side] = isls
                if isls:
                    global_max_island_len[side] = max(global_max_island_len[side], len(isls[0]))

        for obj, sdata in obj_sel_data.items():
            mesh = obj.data
            mw = sdata["mw"]
            edges_list = sdata["edges"]

            vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)
            vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)
            vg_rigid_idx = obj.vertex_groups.find(VG_RIGID)

            # 사용자가 수동 지정해 둔 기존 강체 정점 보존
            existing_rigid = set()
            for v in mesh.vertices:
                for g in v.groups:
                    if g.group == vg_rigid_idx and g.weight > 0.5:
                        existing_rigid.add(v.index)

            all_v_indices = [v.index for v in mesh.vertices]
            vg_l.remove(all_v_indices)
            vg_r.remove(all_v_indices)
            vg_rigid.remove(all_v_indices)

            selected_set = set(sdata["left"]) | set(sdata["right"])
            preserved_rigid = existing_rigid & selected_set
            if preserved_rigid:
                vg_rigid.add(list(preserved_rigid), 1.0, 'REPLACE')

            for side, vlist, vg in (("L", sdata["left"], vg_l), ("R", sdata["right"], vg_r)):
                if not vlist or side not in anchor_info:
                    continue

                cent = anchor_info[side]["centroid"]
                radius = anchor_info[side]["radius"]

                # A) 장신구(독립 메쉬 섬) 자동 감지 (Hair 등 서브 오브젝트에 장신구만 단독으로 있는 경우도 감지)
                rigid_verts = set(v for v in vlist if v in preserved_rigid)
                if props.auto_detect_rigid:
                    islands = obj_side_islands[obj].get(side, [])
                    max_main_len = global_max_island_len[side]
                    for idx_isl, isl in enumerate(islands):
                        is_main_breast = (idx_isl == 0 and len(isl) == max_main_len and len(isl) > props.rigid_max_verts)
                        if not is_main_breast and len(isl) <= props.rigid_max_verts:
                            rigid_verts.update(isl)
                            total_rigid_islands += 1

                # B) 피부 및 옷감 정점 감쇠 베이크 (동일 3D 좌표의 다중 메쉬가 100% 동일한 가중치를 공유하도록 순수 거리 감쇠 유지)
                for vi in vlist:
                    if vi in rigid_verts:
                        continue
                    p_w = mw @ mesh.vertices[vi].co
                    dist_norm = (p_w - cent).length / max(0.02, radius * 1.5)
                    w = max(0.0, min(1.0, 1.0 - dist_norm))
                    w = math.pow(w, falloff_p)
                    vg.add([vi], w, 'REPLACE')

                if rigid_verts:
                    vg_rigid.add(list(rigid_verts), 1.0, 'REPLACE')

        # C) 다중 오브젝트(Body 케이스 + Hair 악세서리 등) 간 3D 공간 통합 강체 클러스터 가중치 동기화
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
    3D 월드 공간 위치 기준으로 하나의 '통합 강체 클러스터(Cross-Object Rigid Cluster)'로 묶어
    완벽히 동일한 단일 (w_l, w_r) 가중치를 부여하고, 강체 바로 밑/주변 케이스 정점(6mm 이내)도
    동일 가중치로 동기화하여 악세서리가 케이스를 뚫고 나오는 현상을 원천 방지합니다.
    """
    if not target_objs or not bone_info:
        return 0

    all_islands = []  # list of dicts: {"obj": obj, "verts": isl, "world_pts": [...], "bmin": Vector, "bmax": Vector, "cent": Vector}

    for obj in target_objs:
        if obj.type != 'MESH':
            continue
        mesh = obj.data
        mw = obj.matrix_world
        vg_rigid_idx = obj.vertex_groups.find(VG_RIGID)
        if vg_rigid_idx < 0:
            continue

        rigid_v_list = []
        for v in mesh.vertices:
            for g in v.groups:
                if g.group == vg_rigid_idx and g.weight > 0.5:
                    rigid_v_list.append(v.index)
                    break

        if not rigid_v_list:
            continue

        edges_list = [(e.vertices[0], e.vertices[1]) for e in mesh.edges]
        islands = _find_connected_islands(rigid_v_list, edges_list)
        for isl in islands:
            if not isl:
                continue
            w_pts = [mw @ mesh.vertices[vi].co for vi in isl]
            xs = [p.x for p in w_pts]
            ys = [p.y for p in w_pts]
            zs = [p.z for p in w_pts]
            bmin = Vector((min(xs), min(ys), min(zs)))
            bmax = Vector((max(xs), max(ys), max(zs)))
            cent = sum(w_pts, Vector((0.0, 0.0, 0.0))) / len(w_pts)
            # 클러스터 거리 판별용 샘플 정점 (최대 40개)
            step = max(1, len(w_pts) // 40)
            sample_pts = w_pts[::step]
            all_islands.append({
                "obj": obj,
                "verts": isl,
                "world_pts": w_pts,
                "sample_pts": sample_pts,
                "bmin": bmin,
                "bmax": bmax,
                "cent": cent,
            })

    if not all_islands:
        return 0

    # 3D 바운딩 박스 중첩 및 근접도(2.5cm) 기준으로 다중 오브젝트 강체 섬들을 통합 클러스터로 병합
    n_isl = len(all_islands)
    margin = 0.025
    adj = {i: [] for i in range(n_isl)}

    for i in range(n_isl):
        a = all_islands[i]
        for j in range(i + 1, n_isl):
            b = all_islands[j]
            # AABB + margin 중첩 검사
            if (a["bmin"].x - margin <= b["bmax"].x and a["bmax"].x + margin >= b["bmin"].x and
                a["bmin"].y - margin <= b["bmax"].y and a["bmax"].y + margin >= b["bmin"].y and
                a["bmin"].z - margin <= b["bmax"].z and a["bmax"].z + margin >= b["bmin"].z):
                connected = (a["cent"] - b["cent"]).length <= 0.06
                if not connected:
                    for pa in a["sample_pts"]:
                        if any((pa - pb).length <= margin for pb in b["sample_pts"]):
                            connected = True
                            break
                if connected:
                    adj[i].append(j)
                    adj[j].append(i)

    visited = set()
    clusters = []
    for i in range(n_isl):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(all_islands[curr])
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            clusters.append(comp)

    def _eval_side_weight(pt, side):
        if side not in bone_info:
            return 0.0
        cent = bone_info[side]["centroid"]
        radius = bone_info[side]["radius"]
        dist_norm = (pt - cent).length / max(0.02, radius * 1.5)
        w = math.pow(max(0.0, 1.0 - dist_norm), falloff_p)
        return max(0.15, min(1.0, w))

    # 모든 강체 정점들의 월드 좌표 KDTree 구축 (주변 케이스/소켓 정점 6mm 동기화용)
    total_rigid_pts = sum(len(isl["world_pts"]) for comp in clusters for isl in comp)
    kd_rigid = KDTree(total_rigid_pts) if total_rigid_pts > 0 else None
    kd_cluster_weights = []
    kd_idx = 0

    for comp in clusters:
        cluster_pts = [p for isl in comp for p in isl["world_pts"]]
        c_cent = sum(cluster_pts, Vector((0.0, 0.0, 0.0))) / len(cluster_pts)
        min_x = min(p.x for p in cluster_pts)
        max_x = max(p.x for p in cluster_pts)

        # 중앙선(X=0)에 걸쳐 있는 가슴 중앙 장신구/케이스는 좌/우 가중치를 동일 비율로 묶어 반으로 갈라지지 않게 보호
        if min_x < -0.004 and max_x > 0.004 and "L" in bone_info and "R" in bone_info:
            wl_raw = _eval_side_weight(c_cent, "L")
            wr_raw = _eval_side_weight(c_cent, "R")
            span = max(0.02, max_x - min_x)
            t_l = max(0.0, min(1.0, 0.5 + c_cent.x / span))
            cluster_wl = wl_raw * t_l
            cluster_wr = wr_raw * (1.0 - t_l)
        elif c_cent.x >= 0.0:
            cluster_wl = _eval_side_weight(c_cent, "L") if "L" in bone_info else _eval_side_weight(c_cent, "R")
            cluster_wr = 0.0 if "L" in bone_info else cluster_wl
            if "L" not in bone_info:
                cluster_wl = 0.0
        else:
            cluster_wr = _eval_side_weight(c_cent, "R") if "R" in bone_info else _eval_side_weight(c_cent, "L")
            cluster_wl = 0.0 if "R" in bone_info else cluster_wr
            if "R" not in bone_info:
                cluster_wr = 0.0

        for isl in comp:
            obj = isl["obj"]
            verts = isl["verts"]
            vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)
            vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)

            if cluster_wl > 0.0:
                vg_l.add(verts, cluster_wl, 'REPLACE')
            else:
                vg_l.remove(verts)

            if cluster_wr > 0.0:
                vg_r.add(verts, cluster_wr, 'REPLACE')
            else:
                vg_r.remove(verts)

            vg_rigid.add(verts, 1.0, 'REPLACE')

            if kd_rigid is not None:
                for p in isl["world_pts"]:
                    kd_rigid.insert(p, kd_idx)
                    kd_cluster_weights.append((cluster_wl, cluster_wr))
                    kd_idx += 1

    # 다른 오브젝트(예: Body에 붙어 있는 케이스 테두리/바닥)에서 강체 장신구와 밀착(6mm 이내)된 정점들을
    # 해당 강체 클러스터와 동일한 가중치로 고정하고, 주변 피부(6mm~15mm)를 매끄럽게 보간
    if kd_rigid is not None and kd_idx > 0:
        kd_rigid.balance()
        r_lock = 0.006   # 6mm 이내: 케이스/소켓 정점으로 간주하여 강체와 100% 동일 가중치로 잠금
        r_blend = 0.015  # 6mm ~ 15mm: 주변 피부 찢어짐 방지용 스무스 전이 구간

        for obj in target_objs:
            if obj.type != 'MESH':
                continue
            mesh = obj.data
            mw = obj.matrix_world
            vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)
            vg_rigid = obj.vertex_groups.get(VG_RIGID) or obj.vertex_groups.new(name=VG_RIGID)
            vg_l_idx = obj.vertex_groups.find(VG_BREAST_L)
            vg_r_idx = obj.vertex_groups.find(VG_BREAST_R)
            vg_rigid_idx = obj.vertex_groups.find(VG_RIGID)

            for v in mesh.vertices:
                cur_wl = 0.0
                cur_wr = 0.0
                is_already_rigid = False
                for g in v.groups:
                    if g.group == vg_rigid_idx and g.weight > 0.5:
                        is_already_rigid = True
                        break
                    elif g.group == vg_l_idx:
                        cur_wl = g.weight
                    elif g.group == vg_r_idx:
                        cur_wr = g.weight

                if is_already_rigid or (cur_wl <= 0.0001 and cur_wr <= 0.0001):
                    continue

                p_w = mw @ v.co
                _, nearest_idx, dist = kd_rigid.find(p_w)
                if dist <= r_lock:
                    c_wl, c_wr = kd_cluster_weights[nearest_idx]
                    if c_wl > 0.0:
                        vg_l.add([v.index], c_wl, 'REPLACE')
                    else:
                        vg_l.remove([v.index])
                    if c_wr > 0.0:
                        vg_r.add([v.index], c_wr, 'REPLACE')
                    else:
                        vg_r.remove([v.index])
                    vg_rigid.add([v.index], 1.0, 'REPLACE')
                elif dist < r_blend:
                    c_wl, c_wr = kd_cluster_weights[nearest_idx]
                    u = (dist - r_lock) / (r_blend - r_lock)
                    s = u * u * (3.0 - 2.0 * u)
                    new_wl = c_wl * (1.0 - s) + cur_wl * s
                    new_wr = c_wr * (1.0 - s) + cur_wr * s
                    if new_wl > 0.0001:
                        vg_l.add([v.index], new_wl, 'REPLACE')
                    else:
                        vg_l.remove([v.index])
                    if new_wr > 0.0001:
                        vg_r.add([v.index], new_wr, 'REPLACE')
                    else:
                        vg_r.remove([v.index])

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
        cent = head_w + fwd * 0.4
        tip = tail_w
        radius = max(0.03, db.head_radius, db.envelope_distance / 1.4, fwd.length * 0.5)

        bone_info[side] = {
            "bname": bname,
            "centroid": cent,
            "tip": tip,
            "head": head_w,
            "tail": tail_w,
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
    updated_obj_count = 0
    total_recomputed_verts = 0

    for obj in target_objs:
        ensure_object_visible(context, obj)
        mesh = obj.data
        mw = obj.matrix_world

        vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
        vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)
        vg_l_idx = obj.vertex_groups.find(VG_BREAST_L)
        vg_r_idx = obj.vertex_groups.find(VG_BREAST_R)
        vg_rigid_idx = obj.vertex_groups.find(VG_RIGID)

        l_verts = []
        r_verts = []
        rigid_verts = set()

        for v in mesh.vertices:
            for g in v.groups:
                if g.group == vg_rigid_idx and g.weight > 0.5:
                    rigid_verts.add(v.index)
                if g.group == vg_l_idx and g.weight > 0.0001:
                    l_verts.append(v.index)
                elif g.group == vg_r_idx and g.weight > 0.0001:
                    r_verts.append(v.index)

        # 만약 해당 메쉬에 등록된 가슴/강체 정점이 전혀 없다면, 본 앵커 반경(radius * 1.5) 내의 정점들을 자동 수집
        if not l_verts and not rigid_verts and "L" in bone_info:
            cent_l = bone_info["L"]["centroid"]
            rad_l = bone_info["L"]["radius"]
            for v in mesh.vertices:
                p_w = mw @ v.co
                if p_w.x >= 0.0 and (p_w - cent_l).length <= rad_l * 1.5:
                    l_verts.append(v.index)

        if not r_verts and not rigid_verts and "R" in bone_info:
            cent_r = bone_info["R"]["centroid"]
            rad_r = bone_info["R"]["radius"]
            for v in mesh.vertices:
                p_w = mw @ v.co
                if p_w.x < 0.0 and (p_w - cent_r).length <= rad_r * 1.5:
                    r_verts.append(v.index)

        for side, vlist, vg in (("L", l_verts, vg_l), ("R", r_verts, vg_r)):
            if not vlist or not vg or side not in bone_info:
                continue

            cent = bone_info[side]["centroid"]
            radius = bone_info[side]["radius"]

            for vi in vlist:
                if vi in rigid_verts:
                    continue
                p_w = mw @ mesh.vertices[vi].co
                dist_norm = (p_w - cent).length / max(0.02, radius * 1.5)
                w = max(0.0, min(1.0, 1.0 - dist_norm))
                w = math.pow(w, falloff_p)
                vg.add([vi], w, 'REPLACE')
                total_recomputed_verts += 1

        total_recomputed_verts += len(rigid_verts)
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

        count, msg = _rebake_mesh_weights_from_rig(context, rig, props)
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
