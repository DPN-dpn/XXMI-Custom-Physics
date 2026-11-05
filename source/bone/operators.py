import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import (
    ensure_object_visible,
    ensure_all_objects_in_object_mode,
    read_vertex_group_arrays,
    world_vertex_positions,
    get_target_mesh_objects,
    compute_density_invariant_geometric_center,
)
from .rig import (
    BREAST_RIG_NAME,
    VG_BREAST_L,
    VG_BREAST_R,
    get_or_create_breast_rig,
)
from .weighting import (
    VG_RIGID,
    synchronize_coincident_vertices_across_objects,
    rebake_mesh_weights_from_rig,
)
from .data import (
    get_bone_data,
    set_bone_data,
    get_bone_item,
    ensure_default_breast_bone_items,
)

# 하위 호환용 별칭
_rebake_mesh_weights_from_rig = rebake_mesh_weights_from_rig
_get_or_create_breast_rig = get_or_create_breast_rig


class XXMI_OT_register_breast_selection(bpy.types.Operator):
    bl_idname = "object.xxmi_register_breast_selection"
    bl_label = "1. 모핑 영역 선택 등록"
    bl_description = "에디트 모드에서 선택한 정점들을 모핑 영역으로 등록합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        objs = get_target_mesh_objects(context)
        return len(objs) > 0

    def execute(self, context):
        target_objs = get_target_mesh_objects(context)
        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        for obj in target_objs:
            ensure_object_visible(context, obj)

        total_l = 0
        total_r = 0
        data_l = {}
        data_r = {}

        for obj in target_objs:
            mesh = obj.data
            n_verts = len(mesh.vertices)
            if n_verts == 0:
                continue

            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0]
            if len(sel_indices) == 0:
                continue

            world = world_vertex_positions(obj)

            kd_local = KDTree(n_verts)
            for vi, wpos in enumerate(world):
                kd_local.insert(Vector(wpos), vi)
            kd_local.balance()

            extra_indices = []
            for si in sel_indices.tolist():
                hits = kd_local.find_range(Vector(world[si]), 0.0012)
                for _, hi, _ in hits:
                    if not sel_mask[hi]:
                        sel_mask[hi] = True
                        extra_indices.append(hi)

            if extra_indices:
                sel_indices = np.nonzero(sel_mask)[0]

            vg_l = obj.vertex_groups.get(VG_BREAST_L) or obj.vertex_groups.new(name=VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R) or obj.vertex_groups.new(name=VG_BREAST_R)

            all_v = list(range(n_verts))
            vg_l.remove(all_v)
            vg_r.remove(all_v)

            sel_list = sel_indices.tolist()
            xs = world[sel_indices, 0]
            # 인공적인 X=0 분할선 없이, 등록된 가슴 영역을 통합 후보군으로 등록
            only_left = np.all(xs > 0.02)
            only_right = np.all(xs < -0.02)

            if only_left:
                vg_l.add(sel_list, 0.05, 'REPLACE')
                total_l += len(sel_list)
                data_l[obj.name] = sel_list
            elif only_right:
                vg_r.add(sel_list, 0.05, 'REPLACE')
                total_r += len(sel_list)
                data_r[obj.name] = sel_list
            else:
                # 양쪽 모두 가중치 평가 후보군으로 등록하되, 각 본의 형태 중심 계산용 정점은 좌/우로 적절히 분류
                vg_l.add(sel_list, 0.05, 'REPLACE')
                vg_r.add(sel_list, 0.05, 'REPLACE')
                left_sub = [vi for vi in sel_list if world[vi, 0] >= -0.01]
                right_sub = [vi for vi in sel_list if world[vi, 0] <= 0.01]
                total_l += len(left_sub)
                total_r += len(right_sub)
                data_l[obj.name] = left_sub
                data_r[obj.name] = right_sub

        synchronize_coincident_vertices_across_objects(target_objs, tol=0.0012)

        props = context.scene.xxmi_custom_physics_props
        ensure_default_breast_bone_items(props)
        item_1 = props.bone_items[0] if len(props.bone_items) > 0 else None
        item_2 = props.bone_items[1] if len(props.bone_items) > 1 else None
        if item_1:
            set_bone_data(item_1, data_l)
        if item_2:
            set_bone_data(item_2, data_r)

        total_sel = max(total_l, total_r)
        if total_sel < 6:
            if was_edit:
                for obj in target_objs:
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode='EDIT')
            self.report({'ERROR'}, "에디트 모드에서 가슴 표면 정점들을 먼저 선택해주세요. (최소 6개 이상)")
            return {'CANCELLED'}

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        self.report(
            {'INFO'},
            f"1단계 완료: 모핑 정점 총 {total_sel}개 등록 완료! "
            f"이어서 [2. 모핑 본 자동 생성]을 진행하세요."
        )
        return {'FINISHED'}


class XXMI_OT_generate_breast_rig(bpy.types.Operator):
    bl_idname = "object.xxmi_generate_breast_rig"
    bl_label = "모든 모핑 본 위치/크기 일괄 재배치"
    bl_description = "등록된 모든 모핑 본 슬롯의 정점 3D 기하학적 형태 중심을 정밀 분석하여 물리 본(XXMI_Physics_Rig)을 자동 배치합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        if props and any(it.vert_count > 0 for it in props.bone_items):
            return True
        return any(
            o.type == 'MESH' and any(vg.name.startswith("XXMI_") for vg in o.vertex_groups)
            for o in bpy.data.objects
        )

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props

        if not props.bone_items:
            self.report({'ERROR'}, "등록된 모핑 본 슬롯이 없습니다. 먼저 [+] 버튼을 눌러 슬롯을 생성하세요.")
            return {'CANCELLED'}

        synced_count = 0
        for item in props.bone_items:
            res = sync_slot_bone_to_rig(context, item)
            if res:
                synced_count += 1

        if synced_count == 0:
            self.report({'WARNING'}, "본 슬롯에 할당된 정점이 부족하여 본을 배치하지 못했습니다. 에디트 모드에서 정점을 선택 후 [선택 정점 할당]을 진행하세요.")
            return {'CANCELLED'}

        rig = props.anchor_obj or bpy.data.objects.get(BREAST_RIG_NAME)
        if rig:
            rig.show_in_front = True
            context.view_layer.objects.active = rig
            rig.select_set(True)

        self.report(
            {'INFO'},
            f"총 {synced_count}개 모핑 본 위치 및 크기 배치 완료! 뷰포트에서 본의 위치를 확인하고 필요시 에디트 모드에서 미세 조정한 뒤 3패널에서 가중치를 계산하세요."
        )
        return {'FINISHED'}


class XXMI_OT_rebake_breast_weights(bpy.types.Operator):
    bl_idname = "object.xxmi_rebake_breast_weights"
    bl_label = "본 & 강체 가중치 자동 계산 & 블렌딩"
    bl_description = "모핑 본들을 기준으로 거리 감쇄 가중치를 계산하고 강체(액세서리)를 포함하여 자연스럽게 블렌딩합니다"
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
            self.report({'ERROR'}, "가슴 물리 본을 찾을 수 없습니다. [2. 가슴 물리 본 생성]을 먼저 진행해주세요.")
            return {'CANCELLED'}

        was_mode = context.mode
        active_obj = context.view_layer.objects.active
        ensure_all_objects_in_object_mode(context)

        context.view_layer.update()

        count, msg = rebake_mesh_weights_from_rig(context, rig, props)

        context.view_layer.update()

        if active_obj and active_obj.name in bpy.data.objects:
            try:
                ensure_object_visible(context, active_obj)
                context.view_layer.objects.active = active_obj
                if was_mode in ('EDIT_ARMATURE', 'POSE'):
                    target_mode = 'EDIT' if was_mode == 'EDIT_ARMATURE' else 'POSE'
                    bpy.ops.object.mode_set(mode=target_mode)
                elif was_mode == 'EDIT_MESH':
                    bpy.ops.object.mode_set(mode='EDIT')
            except Exception:
                pass

        if count == 0:
            self.report({'WARNING'}, msg)
            return {'CANCELLED'}

        self.report({'INFO'}, f"가슴 가중치 계산 완료: {msg}")
        return {'FINISHED'}


class XXMI_OT_select_breast_vertices(bpy.types.Operator):
    bl_idname = "object.xxmi_select_breast_vertices"
    bl_label = "가슴 정점 다시 선택"
    bl_description = "등록된 가슴 정점들을 에디트 모드에서 다시 선택합니다"
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
            self.report({'WARNING'}, "등록된 가슴 정점이 없습니다.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        for obj in target_objs:
            ensure_object_visible(context, obj)
            obj.select_set(True)

        context.view_layer.objects.active = target_objs[0]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')

        total_selected = 0
        for obj in target_objs:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            vg_l = obj.vertex_groups.get(VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R)
            g_l = vg_l.index if vg_l else -1
            g_r = vg_r.index if vg_r else -1
            dvert_lay = bm.verts.layers.deform.verify()

            for v in bm.verts:
                dvert = v[dvert_lay]
                has_w = (g_l in dvert and dvert[g_l] > 0.0001) or (g_r in dvert and dvert[g_r] > 0.0001)
                v.select_set(has_w)
                if has_w:
                    total_selected += 1

            bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"총 {total_selected}개 가슴 정점을 다시 선택했습니다.")
        return {'FINISHED'}


def sync_slot_bone_to_rig(context, item):
    """
    지정된 모핑 본 슬롯 item의 정점 데이터를 기반으로
    XXMI_Breast_Rig 내의 해당 본을 생성하거나 위치/크기를 자동 갱신합니다.
    """
    data = get_bone_data(item)
    if not data:
        return None

    # 해당 슬롯의 모든 정점 월드 좌표 수집
    pts = []
    target_objs = []
    for oname, vis in data.items():
        obj = bpy.data.objects.get(oname)
        if obj and obj.type == 'MESH' and vis:
            target_objs.append(obj)
            w_arr = world_vertex_positions(obj)
            for vi in vis:
                if vi < len(w_arr):
                    pts.append(w_arr[vi])

    if len(pts) < 4:
        return None

    props = context.scene.xxmi_custom_physics_props
    ref_obj = target_objs[0]
    rig = get_or_create_breast_rig(context, props, ref_obj)
    ensure_object_visible(context, rig)

    # 정점들의 형태 중심 및 바운딩 박스 계산
    center, bmin, bmax = compute_density_invariant_geometric_center(pts)
    depth = max(0.04, bmax.y - bmin.y)
    span_x = max(0.04, bmax.x - bmin.x)
    span_z = max(0.04, bmax.z - bmin.z)
    bone_radius = max(0.05, span_x * 0.5, span_z * 0.5, depth * 0.75)

    dx_out = 0.05 * center.x if abs(center.x) > 0.02 else 0.0
    head = Vector((center.x, bmax.y, center.z))
    tail = Vector((center.x + dx_out, bmin.y, center.z))
    if (tail - head).length < 0.04:
        tail = Vector((center.x + dx_out, head.y - 0.04, center.z))

    was_mode = context.mode
    active_obj = context.view_layer.objects.active

    if was_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')

    ebones = rig.data.edit_bones
    btype = getattr(item, "bone_type", 'MORPH')

    if btype in {'TAIL', 'HAIR'}:
        # 다관절 체인 (순차 연결 FK 본) 생성
        n_segs = max(1, getattr(item, "chain_segments", 4))
        vec_delta = (tail - head) / n_segs
        prev_bone = None
        for si in range(n_segs):
            s_name = item.name if si == 0 else f"{item.name}.{si:03d}"
            eb = ebones.get(s_name) or ebones.new(s_name)
            s_head = head + vec_delta * si
            s_tail = head + vec_delta * (si + 1)
            eb.head = s_head
            eb.tail = s_tail
            seg_radius = bone_radius * (1.0 - 0.4 * (si / max(1, n_segs)))
            eb.head_radius = seg_radius
            eb.tail_radius = seg_radius * 0.8
            eb.envelope_distance = seg_radius * 1.3
            eb["xxmi_bone_type"] = btype
            if prev_bone:
                eb.parent = prev_bone
                eb.use_connect = True
            prev_bone = eb
        bone_out_name = item.name
    else:
        eb = ebones.get(item.name)
        if not eb:
            eb = ebones.new(item.name)

        eb.head = head
        eb.tail = tail
        eb.head_radius = bone_radius
        eb.tail_radius = bone_radius * 0.7
        eb.envelope_distance = bone_radius * 1.4
        eb["xxmi_bone_type"] = btype

        bone_out_name = eb.name

    bpy.ops.object.mode_set(mode='OBJECT')
    rig.show_in_front = True

    # 원래 모드로 복귀 (만약 에디트 모드였다면 원래 오브젝트 선택)
    if was_mode == 'EDIT_MESH' and active_obj:
        context.view_layer.objects.active = active_obj
        bpy.ops.object.mode_set(mode='EDIT')

    return bone_out_name


def get_all_rigid_verts_by_obj(props):
    """등록된 모든 강체 슬롯에서 강체 정점 목록을 오브젝트별 set으로 수집합니다."""
    rigid_verts_by_obj = {}
    from ..rigid.data import get_rigid_data
    for r_item in props.rigid_items:
        r_dict = get_rigid_data(r_item)
        if not r_dict:
            continue
        for oname, r_vis in r_dict.items():
            rigid_verts_by_obj.setdefault(oname, set()).update(r_vis)
    return rigid_verts_by_obj


def subtract_rigid_from_bone_slot_data(item, rigid_verts_by_obj, keep_boundary=True):
    """
    모핑 본 슬롯 item의 정점 목록에서 등록된 강체 정점들을 제외합니다.
    keep_boundary=True일 경우, 비-강체(가슴/모핑 표면) 정점과 엣지로 직접 연결되어 있거나
    공간상 맞닿아 있는 경계 접합 정점은 제외하지 않고 유지합니다.

    반환값: (subtracted_count, kept_boundary_count)
    """
    data = get_bone_data(item)
    if not data:
        return 0, 0

    new_data = {}
    subtracted_count = 0
    kept_boundary_count = 0

    for obj_name, v_indices in data.items():
        obj = bpy.data.objects.get(obj_name)
        if not obj or obj.type != 'MESH':
            new_data[obj_name] = v_indices
            continue

        rigid_set = rigid_verts_by_obj.get(obj_name, set())
        slot_set = set(v_indices)

        overlap = slot_set & rigid_set
        if not overlap:
            new_data[obj_name] = v_indices
            continue

        mesh = obj.data

        boundary_verts = set()
        if keep_boundary:
            # 1. 엣지 토폴로지 연결 검사: overlap 정점 중 비-강체 정점과 엣지로 연결된 정점 유지
            edge_neighbors = {v: [] for v in overlap}
            for e in mesh.edges:
                v1, v2 = e.vertices[0], e.vertices[1]
                if v1 in edge_neighbors:
                    edge_neighbors[v1].append(v2)
                if v2 in edge_neighbors:
                    edge_neighbors[v2].append(v1)

            for v, nbrs in edge_neighbors.items():
                if any(nbr not in rigid_set for nbr in nbrs):
                    boundary_verts.add(v)

            # 2. 메쉬가 분리되어 있어 엣지가 없는 경우를 위한 공간 근접성(KDTree) 보정
            non_rigid_slot = [v for v in slot_set if v not in rigid_set]
            if non_rigid_slot and (len(boundary_verts) < len(overlap)):
                wpos = world_vertex_positions(obj)
                kd = KDTree(len(non_rigid_slot))
                for idx, vi in enumerate(non_rigid_slot):
                    kd.insert(Vector(wpos[vi]), idx)
                kd.balance()

                for v in overlap:
                    if v not in boundary_verts:
                        pt = Vector(wpos[v])
                        hits = kd.find_range(pt, 0.0015)
                        if hits:
                            boundary_verts.add(v)

        # 최종 남길 정점: (slot_set - overlap) + boundary_verts
        to_remove = overlap - boundary_verts
        remaining = [v for v in v_indices if v not in to_remove]

        new_data[obj_name] = remaining
        subtracted_count += len(to_remove)
        kept_boundary_count += len(boundary_verts)

        # 오브젝트의 버텍스 그룹에서도 제거된 정점 제외
        vg_target_name = f"XXMI_{item.name}"
        vg = obj.vertex_groups.get(vg_target_name)
        if vg and to_remove:
            vg.remove(list(to_remove))

    set_bone_data(item, new_data)
    return subtracted_count, kept_boundary_count


class XXMI_OT_bone_item_add(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_add"
    bl_label = "물리 본 슬롯 추가"
    bl_description = "새로운 물리 본 슬롯을 추가합니다 (버텍스 선택 상태에서 누르면 즉시 할당)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items.add()

        # 이름 및 종류 설정
        n_items = len(props.bone_items)
        selected_type = getattr(props, "bone_type", 'MORPH')
        item.bone_type = selected_type

        type_names = {
            'MORPH': "모핑 본",
            'TAIL': "꼬리 본",
            'HAIR': "헤어 본",
            'SKIRT': "스커트 본",
            'WING': "날개 본",
        }
        type_prefix = type_names.get(selected_type, "물리 본")
        item.name = f"{type_prefix} {n_items}"

        item.peak_weight = getattr(props, "peak_weight", 0.40)
        item.falloff_power = getattr(props, "falloff_power", 1.0)
        item.smooth_iterations = getattr(props, "smooth_iterations", 2)
        item.lock_registered_verts = getattr(props, "lock_registered_verts", True)
        item.strength = getattr(props, "strength", 1.0)
        item.speed = getattr(props, "speed", 1.1)
        item.damping = getattr(props, "damping", 0.85)
        item.inertia = getattr(props, "inertia", 1.2)
        item.idle_motion = getattr(props, "idle_motion", 0.4)

        props.bone_item_index = n_items - 1

        # 현재 에디트 모드에서 선택된 정점이 있다면 즉시 할당
        was_edit = (context.mode == 'EDIT_MESH')
        target_objs = get_target_mesh_objects(context)
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        data = {}
        vg_target_name = f"XXMI_{item.name}"
        if target_objs:
            for obj in target_objs:
                mesh = obj.data
                n_verts = len(mesh.vertices)
                if n_verts == 0:
                    continue
                sel_mask = np.zeros(n_verts, dtype=bool)
                mesh.vertices.foreach_get("select", sel_mask)
                sel_indices = np.nonzero(sel_mask)[0].tolist()
                if sel_indices:
                    data[obj.name] = sel_indices
                    vg = obj.vertex_groups.get(vg_target_name) or obj.vertex_groups.new(name=vg_target_name)
                    vg.add(sel_indices, 0.05, 'REPLACE')

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if data:
            set_bone_data(item, data)
            sync_slot_bone_to_rig(context, item)
            self.report({'INFO'}, f"새 물리 본 슬롯 '{item.name}' 생성 및 정점 {item.vert_count}개 할당 / 본 생성 완료!")
        else:
            self.report({'INFO'}, f"새 물리 본 슬롯 '{item.name}'이 추가되었습니다.")

        return {'FINISHED'}


class XXMI_OT_bone_item_remove(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_remove"
    bl_label = "물리 본 슬롯 삭제"
    bl_description = "선택된 물리 본 슬롯을 목록에서 삭제합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.bone_item_index < len(props.bone_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        idx = props.bone_item_index
        item = props.bone_items[idx]
        name = item.name

        # 리그에서 해당 본 삭제
        rig = props.anchor_obj or bpy.data.objects.get(BREAST_RIG_NAME)
        if rig and rig.type == 'ARMATURE':
            was_mode = context.mode
            if was_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode='EDIT')
            if name in rig.data.edit_bones:
                rig.data.edit_bones.remove(rig.data.edit_bones[name])
            bpy.ops.object.mode_set(mode='OBJECT')

        props.bone_items.remove(idx)
        props.bone_item_index = max(0, min(idx, len(props.bone_items) - 1))
        self.report({'INFO'}, f"물리 본 슬롯 '{name}'을 삭제했습니다.")
        return {'FINISHED'}


class XXMI_OT_bone_item_assign(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_assign"
    bl_label = "선택 정점 이 본에 할당"
    bl_description = "에디트 모드에서 현재 선택된 정점들을 이 물리 본 슬롯에 할당합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.bone_item_index < len(props.bone_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items[props.bone_item_index]
        data = get_bone_data(item)

        target_objs = get_target_mesh_objects(context)
        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        assigned_any = False
        vg_target_name = f"XXMI_{item.name}"

        for obj in target_objs:
            mesh = obj.data
            n_verts = len(mesh.vertices)
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0].tolist()
            if not sel_indices:
                continue

            existing = set(data.get(obj.name, []))
            existing.update(sel_indices)
            data[obj.name] = sorted(list(existing))
            assigned_any = True

            vg = obj.vertex_groups.get(vg_target_name) or obj.vertex_groups.new(name=vg_target_name)
            vg.add(sel_indices, 0.05, 'REPLACE')

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if not assigned_any:
            self.report({'WARNING'}, "에디트 모드에서 본에 할당할 정점을 먼저 선택해주세요.")
            return {'CANCELLED'}

        set_bone_data(item, data)
        sync_slot_bone_to_rig(context, item)
        self.report({'INFO'}, f"본 '{item.name}'에 정점 할당 및 본 위치 갱신 완료 (총 {item.vert_count}개).")
        return {'FINISHED'}


class XXMI_OT_bone_item_subtract_rigid_verts(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_subtract_rigid_verts"
    bl_label = "강체 정점 자동 제외 (경계 유지)"
    bl_description = "현재 모핑 본 슬롯에서 등록된 강체 정점들을 자동 제외합니다 (모핑 표면과 맞닿아 있는 접합부 경계 버텍스는 자연스러운 연결을 위해 유지)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return (0 <= props.bone_item_index < len(props.bone_items)) and len(props.rigid_items) > 0

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items[props.bone_item_index]

        rigid_verts_by_obj = get_all_rigid_verts_by_obj(props)
        if not rigid_verts_by_obj:
            self.report({'WARNING'}, "등록된 강체 정점이 없습니다. 먼저 [2. 강체(액세서리) 설정]에서 강체를 등록해주세요.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        subtracted, kept = subtract_rigid_from_bone_slot_data(item, rigid_verts_by_obj, keep_boundary=True)

        # 강체 제외 후 순수 모핑 영역에 맞춰 본 위치 갱신
        sync_slot_bone_to_rig(context, item)

        if was_edit:
            target_objs = get_target_mesh_objects(context)
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if subtracted == 0:
            self.report({'INFO'}, f"'{item.name}' 슬롯에 제외할 강체 정점이 없거나 모두 경계 정점으로 보존되었습니다.")
        else:
            self.report({'INFO'}, f"'{item.name}' 슬롯에서 강체 정점 {subtracted}개 제외 완료 (경계 정점 {kept}개 유지, 남은 정점 {item.vert_count}개).")

        return {'FINISHED'}


class XXMI_OT_subtract_rigid_from_all_bones(bpy.types.Operator):
    bl_idname = "object.xxmi_subtract_rigid_from_all_bones"
    bl_label = "모든 모핑 본에서 강체 정점 자동 제외 (경계 유지)"
    bl_description = "모든 모핑 본 슬롯에서 등록된 강체 정점들을 일괄 제외합니다 (표면 접합부 경계 정점 보존)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return len(props.bone_items) > 0 and len(props.rigid_items) > 0

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        rigid_verts_by_obj = get_all_rigid_verts_by_obj(props)
        if not rigid_verts_by_obj:
            self.report({'WARNING'}, "등록된 강체 정점이 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        total_sub = 0
        total_kept = 0
        for item in props.bone_items:
            sub, kept = subtract_rigid_from_bone_slot_data(item, rigid_verts_by_obj, keep_boundary=True)
            total_sub += sub
            total_kept += kept
            sync_slot_bone_to_rig(context, item)

        if was_edit:
            target_objs = get_target_mesh_objects(context)
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        self.report({'INFO'}, f"전체 모핑 본에서 강체 정점 {total_sub}개 제외 완료 (경계 정점 {total_kept}개 유지).")
        return {'FINISHED'}


class XXMI_OT_bone_item_select_verts(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_select_verts"
    bl_label = "정점 확인"
    bl_description = "이 본에 할당된 정점들을 에디트 모드에서 선택하여 확인합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.bone_item_index < len(props.bone_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items[props.bone_item_index]
        data = get_bone_data(item)

        if not data:
            self.report({'WARNING'}, f"본 '{item.name}'에 등록된 정점이 없습니다.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        first_obj = None

        for obj_name, v_indices in data.items():
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.type == 'MESH':
                ensure_object_visible(context, obj)
                obj.select_set(True)
                if not first_obj:
                    first_obj = obj

        if not first_obj:
            self.report({'ERROR'}, "해당 정점을 포함하는 메쉬 오브젝트를 찾을 수 없습니다.")
            return {'CANCELLED'}

        context.view_layer.objects.active = first_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')

        total_selected = 0
        for obj_name, v_indices in data.items():
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.type == 'MESH':
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                v_set = set(v_indices)
                for vi in v_set:
                    if vi < len(bm.verts):
                        bm.verts[vi].select_set(True)
                        total_selected += 1
                bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"본 '{item.name}'의 정점 {total_selected}개를 선택했습니다.")
        return {'FINISHED'}


class XXMI_OT_bone_item_remove_verts(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_remove_verts"
    bl_label = "선택 정점 제외"
    bl_description = "현재 선택된 정점들을 이 물리 본 슬롯에서 제외합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return 0 <= props.bone_item_index < len(props.bone_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items[props.bone_item_index]
        data = get_bone_data(item)

        target_objs = get_target_mesh_objects(context)
        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        removed_any = False
        vg_target_name = f"XXMI_{item.name}"

        for obj in target_objs:
            if obj.name not in data:
                continue
            mesh = obj.data
            n_verts = len(mesh.vertices)
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = set(np.nonzero(sel_mask)[0].tolist())
            if not sel_indices:
                continue

            existing = set(data[obj.name])
            before_len = len(existing)
            existing.difference_update(sel_indices)
            if len(existing) < before_len:
                data[obj.name] = sorted(list(existing))
                removed_any = True
                vg = obj.vertex_groups.get(vg_target_name)
                if vg:
                    vg.remove(list(sel_indices))

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        if not removed_any:
            self.report({'WARNING'}, "제외할 정점이 선택되지 않았습니다.")
            return {'CANCELLED'}

        data = {k: v for k, v in data.items() if v}
        set_bone_data(item, data)
        self.report({'INFO'}, f"본 '{item.name}'에서 정점 제외 완료 (남은 정점 {item.vert_count}개).")
        return {'FINISHED'}


class XXMI_OT_copy_bone_weights_to_all(bpy.types.Operator):
    bl_idname = "object.xxmi_copy_bone_weights_to_all"
    bl_label = "가중치 설정 다른 본에 복사"
    bl_description = "현재 선택된 본의 가중치 설정(최대 가중치, 감쇠, 스무딩 등)을 모든 본에 동일하게 복사합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return len(props.bone_items) > 1 and (0 <= props.bone_item_index < len(props.bone_items))

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        curr = props.bone_items[props.bone_item_index]

        count = 0
        for i, item in enumerate(props.bone_items):
            if i != props.bone_item_index:
                item.peak_weight = curr.peak_weight
                item.falloff_power = curr.falloff_power
                item.smooth_iterations = curr.smooth_iterations
                item.lock_registered_verts = curr.lock_registered_verts
                count += 1

        self.report({'INFO'}, f"본 '{curr.name}'의 가중치 설정을 {count}개 본에 복사했습니다.")
        return {'FINISHED'}


class XXMI_OT_clear_breast_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_clear_breast_physics"
    bl_label = "모핑 본 & 물리 전체 초기화"
    bl_description = "생성된 물리 본과 모핑 가중치 설정을 모두 지우고 초기화합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and (props.anchor_obj is not None or bpy.data.objects.get(BREAST_RIG_NAME) is not None or len(props.bone_items) > 0)

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        from ..simulation.engine import stop_physics_preview
        props = context.scene.xxmi_custom_physics_props
        if props.is_preview_running:
            stop_physics_preview(context)

        for r_name in (BREAST_RIG_NAME, "XXMI_Physics_Rig", "XXMI_Breast_Rig"):
            rig = bpy.data.objects.get(r_name)
            if rig:
                bpy.data.objects.remove(rig, do_unlink=True)
        props.anchor_obj = None

        bone_names = [it.name for it in props.bone_items]
        target_vgs = [f"XXMI_{bn}" for bn in bone_names] + [VG_BREAST_L, VG_BREAST_R, VG_RIGID, "XXMI_Physics"]

        cleared_objs = 0
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                removed_any = False
                for vg in list(obj.vertex_groups):
                    if vg.name.startswith("XXMI_") or vg.name in target_vgs:
                        obj.vertex_groups.remove(vg)
                        removed_any = True
                if removed_any:
                    cleared_objs += 1

        props.bone_items.clear()
        props.bone_item_index = 0

        self.report({'INFO'}, f"모핑 물리 설정 초기화 완료 ({cleared_objs}개 메쉬 오브젝트 가중치 제거됨).")
        return {'FINISHED'}


classes = (
    XXMI_OT_register_breast_selection,
    XXMI_OT_generate_breast_rig,
    XXMI_OT_rebake_breast_weights,
    XXMI_OT_bone_item_add,
    XXMI_OT_bone_item_remove,
    XXMI_OT_bone_item_assign,
    XXMI_OT_bone_item_select_verts,
    XXMI_OT_bone_item_remove_verts,
    XXMI_OT_bone_item_subtract_rigid_verts,
    XXMI_OT_subtract_rigid_from_all_bones,
    XXMI_OT_copy_bone_weights_to_all,
    XXMI_OT_select_breast_vertices,
    XXMI_OT_clear_breast_physics,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
