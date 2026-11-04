import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import (
    ensure_object_visible,
    read_vertex_group_arrays,
    world_vertex_positions,
    get_target_mesh_objects,
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
    bl_label = "1. 가슴 영역 선택 등록"
    bl_description = "에디트 모드에서 선택한 정점들을 가슴 영역으로 등록합니다"
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
            bpy.ops.object.mode_set(mode='OBJECT')

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

            is_left = world[sel_indices, 0] >= 0.0
            left_idx = sel_indices[is_left].tolist()
            right_idx = sel_indices[~is_left].tolist()

            n_l = len(left_idx)
            n_r = len(right_idx)
            if n_l > 0 and n_r > 0:
                if n_l < 25 and n_l < 0.12 * n_r:
                    right_idx = sel_indices.tolist()
                    left_idx = []
                elif n_r < 25 and n_r < 0.12 * n_l:
                    left_idx = sel_indices.tolist()
                    right_idx = []

            if left_idx:
                vg_l.add(left_idx, 0.05, 'REPLACE')
                total_l += len(left_idx)
                data_l[obj.name] = left_idx
            if right_idx:
                vg_r.add(right_idx, 0.05, 'REPLACE')
                total_r += len(right_idx)
                data_r[obj.name] = right_idx

        synchronize_coincident_vertices_across_objects(target_objs, tol=0.0012)

        props = context.scene.xxmi_custom_physics_props
        ensure_default_breast_bone_items(props)
        item_l = get_bone_item(props, "Breast.L")
        item_r = get_bone_item(props, "Breast.R")
        if item_l:
            set_bone_data(item_l, data_l)
        if item_r:
            set_bone_data(item_r, data_r)

        total_sel = total_l + total_r
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
            f"1단계 완료: 가슴 정점 총 {total_sel}개(좌: {total_l}개, 우: {total_r}개) 등록 완료! "
            f"이어서 [2. 가슴 물리 본 생성]을 진행하세요."
        )
        return {'FINISHED'}


class XXMI_OT_generate_breast_rig(bpy.types.Operator):
    bl_idname = "object.xxmi_generate_breast_rig"
    bl_label = "2. 가슴 물리 본 생성"
    bl_description = "등록된 가슴 정점 형태를 분석하여 알맞은 위치와 방향의 가슴 물리 본(Armature)을 자동 생성합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(
            o.type == 'MESH' and (o.vertex_groups.get(VG_BREAST_L) or o.vertex_groups.get(VG_BREAST_R))
            for o in bpy.data.objects
        )

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        target_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and (o.vertex_groups.get(VG_BREAST_L) or o.vertex_groups.get(VG_BREAST_R))
        ]

        if not target_objs:
            self.report({'ERROR'}, "등록된 가슴 정점이 없습니다. 먼저 [1. 가슴 영역 선택 등록]을 진행해주세요.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        for obj in target_objs:
            ensure_object_visible(context, obj)

        all_left_pts = []
        all_right_pts = []
        soft_left_pts = []
        soft_right_pts = []
        total_rigid_count = 0

        for obj in target_objs:
            vg_l = obj.vertex_groups.get(VG_BREAST_L)
            vg_r = obj.vertex_groups.get(VG_BREAST_R)
            vg_rigid = obj.vertex_groups.get(VG_RIGID)

            group_map = {}
            if vg_l: group_map[vg_l.index] = "l"
            if vg_r: group_map[vg_r.index] = "r"
            if vg_rigid: group_map[vg_rigid.index] = "rigid"

            arrs = read_vertex_group_arrays(obj, group_map)
            world = world_vertex_positions(obj)

            rigid_mask = (arrs["rigid"] > 0.5) if "rigid" in arrs else np.zeros(len(world), dtype=bool)
            total_rigid_count += int(np.count_nonzero(rigid_mask))

            if "l" in arrs:
                l_mask = arrs["l"] > 0.0001
                l_idx = np.nonzero(l_mask)[0]
                for vi in l_idx.tolist():
                    pt = Vector(world[vi])
                    all_left_pts.append(pt)
                    if not rigid_mask[vi]:
                        soft_left_pts.append(pt)

            if "r" in arrs:
                r_mask = arrs["r"] > 0.0001
                r_idx = np.nonzero(r_mask)[0]
                for vi in r_idx.tolist():
                    pt = Vector(world[vi])
                    all_right_pts.append(pt)
                    if not rigid_mask[vi]:
                        soft_right_pts.append(pt)

        final_left = soft_left_pts if soft_left_pts else all_left_pts
        final_right = soft_right_pts if soft_right_pts else all_right_pts

        if not final_left and not final_right:
            self.report({'ERROR'}, "유효한 가슴 정점을 찾을 수 없습니다. [1. 가슴 영역 선택 등록]을 다시 진행해주세요.")
            return {'CANCELLED'}

        anchor_info = {}
        total_registered = len(final_left) + len(final_right)

        for side, pts, bname in (("L", final_left, "Breast.L"), ("R", final_right, "Breast.R")):
            n_pts = len(pts)
            if n_pts < 30 or (total_registered > 100 and n_pts < total_registered * 0.10):
                continue

            pts_arr = np.array([p[:] for p in pts], dtype=np.float64)

            xs_sorted = np.sort(pts_arr[:, 0])
            zs_sorted = np.sort(pts_arr[:, 2])
            q1, q3 = int(n_pts * 0.20), int(n_pts * 0.80)
            x_center = float(np.mean(xs_sorted[q1:q3])) if q3 > q1 else float(np.mean(xs_sorted))
            z_center = float(np.mean(zs_sorted[q1:q3])) if q3 > q1 else float(np.mean(zs_sorted))

            ys_sorted = np.sort(pts_arr[:, 1])
            y_min = float(ys_sorted[0])
            y_base = float(ys_sorted[int(n_pts * 0.85)])
            depth = max(0.04, y_base - y_min)

            head_pos = Vector((x_center, y_base, z_center))

            outward_sign = 1.0 if side == 'L' else -1.0
            dx_natural = outward_sign * depth * 0.10
            dz_natural = -depth * 0.05

            front_mask = pts_arr[:, 1] <= (y_min + depth * 0.25)
            front_pts = pts_arr[front_mask]
            if len(front_pts) >= 6:
                fx = float(np.mean(front_pts[:, 0]))
                fz = float(np.mean(front_pts[:, 2]))
                dx_apex = fx - x_center
                dz_apex = fz - z_center
                max_dev = depth * 0.35
                dx_apex = max(-max_dev, min(max_dev, dx_apex))
                dz_apex = max(-max_dev, min(max_dev, dz_apex))
                dx = 0.6 * dx_apex + 0.4 * dx_natural
                dz = 0.6 * dz_apex + 0.4 * dz_natural
            else:
                dx = dx_natural
                dz = dz_natural

            tail_pos = Vector((head_pos.x + dx, head_pos.y - depth, head_pos.z + dz))
            bone_vec_v = tail_pos - head_pos
            bone_len_v = bone_vec_v.length
            bone_u_v = bone_vec_v.normalized() if bone_len_v > 1e-4 else Vector((0.0, -1.0, 0.0))

            x_span = float(xs_sorted[-1] - xs_sorted[0])
            z_span = float(zs_sorted[-1] - zs_sorted[0])
            radius = max(0.03, x_span * 0.5, z_span * 0.5)

            centroid = Vector((x_center, (y_base + y_min) * 0.5, z_center))
            anchor_info[side] = {
                "bname": bname,
                "centroid": centroid,
                "tip": tail_pos,
                "head": head_pos,
                "tail": tail_pos,
                "fwd_unit": bone_u_v,
                "radius": radius,
            }

        if not anchor_info:
            self.report({'ERROR'}, "충분한 수의 가슴 정점을 찾을 수 없습니다. [1. 가슴 영역 선택 등록]을 다시 진행해주세요.")
            return {'CANCELLED'}

        ref_obj = target_objs[0]
        rig = get_or_create_breast_rig(context, props, ref_obj)
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
        rig.show_in_front = True
        context.view_layer.objects.active = rig
        rig.select_set(True)

        self.report(
            {'INFO'},
            f"2단계 완료: 물리 본 생성 완료! 뷰포트에서 본의 위치/각도를 확인하고 [3. 본 기준 가중치(웨이트) 자동 계산]을 누르세요."
        )
        return {'FINISHED'}


class XXMI_OT_rebake_breast_weights(bpy.types.Operator):
    bl_idname = "object.xxmi_rebake_breast_weights"
    bl_label = "3. 본 기준 가중치(웨이트) 자동 계산"
    bl_description = "가슴 물리 본을 기준으로 자연스러운 움직임 가중치를 계산하여 적용합니다"
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
        if was_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        context.view_layer.update()

        count, msg = rebake_mesh_weights_from_rig(context, rig, props)

        context.view_layer.update()

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


class XXMI_OT_bone_item_add(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_add"
    bl_label = "물리 본 슬롯 추가"
    bl_description = "새로운 물리 본 슬롯을 추가합니다 (버텍스 선택 상태에서 누르면 즉시 할당)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items.add()

        # 이름 및 종류 자동 설정
        n_items = len(props.bone_items)
        has_l = any(it.bone_type == 'BREAST_L' or it.name == 'Breast.L' for it in props.bone_items[:-1])
        has_r = any(it.bone_type == 'BREAST_R' or it.name == 'Breast.R' for it in props.bone_items[:-1])

        if not has_l:
            item.name = "Breast.L"
            item.bone_type = 'BREAST_L'
        elif not has_r:
            item.name = "Breast.R"
            item.bone_type = 'BREAST_R'
        else:
            item.name = f"Bone_{n_items}"
            item.bone_type = 'CUSTOM'

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
            bpy.ops.object.mode_set(mode='OBJECT')

        data = {}
        vg_target_name = VG_BREAST_L if item.bone_type == 'BREAST_L' else (VG_BREAST_R if item.bone_type == 'BREAST_R' else f"XXMI_{item.name}")
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
            self.report({'INFO'}, f"새 물리 본 슬롯 '{item.name}' 생성 및 정점 {item.vert_count}개 할당 완료!")
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
            bpy.ops.object.mode_set(mode='OBJECT')

        assigned_any = False
        vg_target_name = VG_BREAST_L if item.bone_type == 'BREAST_L' else (VG_BREAST_R if item.bone_type == 'BREAST_R' else f"XXMI_{item.name}")

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
        self.report({'INFO'}, f"본 '{item.name}'에 정점 할당 완료 (총 {item.vert_count}개).")
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
            bpy.ops.object.mode_set(mode='OBJECT')

        removed_any = False
        vg_target_name = VG_BREAST_L if item.bone_type == 'BREAST_L' else (VG_BREAST_R if item.bone_type == 'BREAST_R' else f"XXMI_{item.name}")

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
    bl_label = "가슴 물리 설정 전체 초기화"
    bl_description = "생성된 물리 본과 가슴 가중치 설정을 모두 지우고 초기화합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = context.scene.xxmi_custom_physics_props
        return props.anchor_obj is not None or bpy.data.objects.get(BREAST_RIG_NAME) is not None

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        from ..simulation.engine import stop_physics_preview
        props = context.scene.xxmi_custom_physics_props
        if props.is_preview_running:
            stop_physics_preview(context)

        rig = props.anchor_obj or bpy.data.objects.get(BREAST_RIG_NAME)
        if rig:
            bpy.data.objects.remove(rig, do_unlink=True)
            props.anchor_obj = None

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
    XXMI_OT_generate_breast_rig,
    XXMI_OT_rebake_breast_weights,
    XXMI_OT_bone_item_add,
    XXMI_OT_bone_item_remove,
    XXMI_OT_bone_item_assign,
    XXMI_OT_bone_item_select_verts,
    XXMI_OT_bone_item_remove_verts,
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
