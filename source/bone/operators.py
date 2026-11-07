import bpy
import bmesh
import numpy as np
from ..utils import (
    ensure_object_visible,
    ensure_all_objects_in_object_mode,
    get_target_mesh_objects,
    world_vertex_positions,
    compute_density_invariant_geometric_center,
)
from .rig import (
    PHYSICS_RIG_NAME,
    BREAST_RIG_NAME,
    VG_BREAST_L,
    VG_BREAST_R,
    get_or_create_physics_rig,
    sync_slot_bone_to_rig,
    reposition_all_slot_bones,
)
from .data import (
    get_bone_data,
    set_bone_data,
    get_all_rigid_verts_by_obj,
    subtract_rigid_from_bone_slot_data,
)
from .weighting import (
    VG_RIGID,
    rebake_mesh_weights_from_rig,
)

# 하위 호환용 별칭
_rebake_mesh_weights_from_rig = rebake_mesh_weights_from_rig
_get_or_create_breast_rig = get_or_create_physics_rig


class XXMI_OT_bone_item_add(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_add"
    bl_label = "물리 본 슬롯 추가"
    bl_description = "새로운 물리 본 슬롯을 추가합니다 (버텍스 선택 상태에서 누르면 즉시 할당 및 리그 본 배치)"
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
        item.prev_name = item.name

        item.radius_scale = getattr(props, "radius_scale", 1.15)
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


class XXMI_OT_bone_item_add_split_lr(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_add_split_lr"
    bl_label = "선택 영역 절반 분할하여 본 2개 추가"
    bl_description = "가슴 전체 등 좌/우가 합쳐진 선택 영역을 중앙(X축) 기준으로 절반으로 나누어 독립된 물리 본 슬롯 2개를 한 번에 생성하고 배치합니다"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        was_edit = (context.mode == 'EDIT_MESH')
        target_objs = get_target_mesh_objects(context)
        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        if was_edit:
            ensure_all_objects_in_object_mode(context)

        # 1. 선택된 모든 정점 및 3D 월드 좌표 수집
        obj_sel_map = {}
        all_sel_pts = []
        for obj in target_objs:
            mesh = obj.data
            n_verts = len(mesh.vertices)
            if n_verts == 0:
                continue
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0].tolist()
            if sel_indices:
                w_arr = world_vertex_positions(obj)
                obj_sel_map[obj] = (sel_indices, w_arr)
                for vi in sel_indices:
                    if vi < len(w_arr):
                        all_sel_pts.append(w_arr[vi])

        if len(all_sel_pts) < 8:
            if was_edit:
                for obj in target_objs:
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, "에디트 모드에서 좌/우로 분할할 영역(예: 가슴 전체)의 정점들을 먼저 선택해주세요.")
            return {'CANCELLED'}

        # 2. 선택 영역의 수평(X축) 중앙 분할 기준선 산출
        center, bmin, bmax = compute_density_invariant_geometric_center(all_sel_pts)
        if bmin.x < -0.01 and bmax.x > 0.01 and abs(center.x) < 0.03:
            x_mid = 0.0
        else:
            x_mid = float(center.x)

        # 중앙 경계선(0.5mm 이내) 정점은 양쪽 모두에 포함시켜 빈틈 방지
        seam_margin = 0.0005
        data_l = {}
        data_r = {}
        for obj, (sel_indices, w_arr) in obj_sel_map.items():
            vis_l = [vi for vi in sel_indices if vi < len(w_arr) and w_arr[vi][0] >= x_mid - seam_margin]
            vis_r = [vi for vi in sel_indices if vi < len(w_arr) and w_arr[vi][0] <= x_mid + seam_margin]
            if vis_l:
                data_l[obj.name] = vis_l
            if vis_r:
                data_r[obj.name] = vis_r

        if not data_l or not data_r:
            if was_edit:
                for obj in target_objs:
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, "선택 영역을 좌/우 절반으로 나눌 수 없습니다. 양쪽 영역이 모두 포함되도록 선택해주세요.")
            return {'CANCELLED'}

        selected_type = getattr(props, "bone_type", 'MORPH')
        type_names = {
            'MORPH': "모핑 본",
            'TAIL': "꼬리 본",
            'HAIR': "헤어 본",
            'SKIRT': "스커트 본",
            'WING': "날개 본",
        }
        type_prefix = type_names.get(selected_type, "물리 본")

        def _create_independent_slot(slot_data):
            item = props.bone_items.add()
            idx_num = len(props.bone_items)
            item.bone_type = selected_type
            item.name = f"{type_prefix} {idx_num}"
            item.prev_name = item.name
            item.radius_scale = getattr(props, "radius_scale", 1.15)
            item.peak_weight = getattr(props, "peak_weight", 0.40)
            item.falloff_power = getattr(props, "falloff_power", 1.0)
            item.smooth_iterations = getattr(props, "smooth_iterations", 2)
            item.lock_registered_verts = getattr(props, "lock_registered_verts", True)
            item.strength = getattr(props, "strength", 1.0)
            item.speed = getattr(props, "speed", 1.1)
            item.damping = getattr(props, "damping", 0.85)
            item.inertia = getattr(props, "inertia", 1.2)
            item.idle_motion = getattr(props, "idle_motion", 0.4)

            vg_target_name = f"XXMI_{item.name}"
            for oname, vis in slot_data.items():
                o = bpy.data.objects.get(oname)
                if o and o.type == 'MESH' and vis:
                    vg = o.vertex_groups.get(vg_target_name) or o.vertex_groups.new(name=vg_target_name)
                    vg.add(vis, 0.05, 'REPLACE')

            set_bone_data(item, slot_data)
            sync_slot_bone_to_rig(context, item)
            return item

        item_l = _create_independent_slot(data_l)
        item_r = _create_independent_slot(data_r)
        props.bone_item_index = len(props.bone_items) - 1

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        self.report(
            {'INFO'},
            f"선택 영역을 절반으로 나누어 독립 본 '{item_l.name}'({item_l.vert_count}개), '{item_r.name}'({item_r.vert_count}개) 2개를 추가했습니다!"
        )
        return {'FINISHED'}


def find_and_remove_bones_from_rigs(context, props, item, idx):
    """
    지정된 물리 본 슬롯(item)에 대응하는 블렌더 리그(Armature) 내의 본들을 찾아 삭제합니다.
    이름 불일치(모핑 본 1 <-> Breast.L 등), 하위 체인 본(.001 등), 대소문자,
    접두사 차이, 슬롯 인덱스 매칭을 모두 지원합니다.
    """
    name = item.name.strip()
    prev_name = getattr(item, "prev_name", "").strip()

    # 1. 씬 내의 모든 물리 리그 후보 검색
    rig_candidates = []
    if getattr(props, "anchor_obj", None):
        rig_candidates.append(props.anchor_obj)
    for r_name in (PHYSICS_RIG_NAME, "XXMI_Breast_Rig"):
        r_obj = bpy.data.objects.get(r_name)
        if r_obj and r_obj not in rig_candidates:
            rig_candidates.append(r_obj)
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and obj not in rig_candidates:
            oname_l = obj.name.lower()
            if "xxmi" in oname_l or "physics" in oname_l or "breast" in oname_l:
                rig_candidates.append(obj)

    # 2. 본 이름 매칭 후보 목록 생성
    cand_names = set()
    if name:
        cand_names.add(name)
        cand_names.add(name.lower())
        cand_names.add(name.replace(".", "_"))
        cand_names.add(name.replace("_", "."))
    if prev_name:
        cand_names.add(prev_name)
        cand_names.add(prev_name.lower())
        cand_names.add(prev_name.replace(".", "_"))
        cand_names.add(prev_name.replace("_", "."))

    # 왼쪽/오른쪽 패턴 및 번호 매칭
    is_left = ("왼" in name) or (".l" in name.lower()) or ("_l" in name.lower()) or (idx == 0 and "1" in name)
    is_right = ("오" in name) or (".r" in name.lower()) or ("_r" in name.lower()) or (idx == 1 and "2" in name)
    if is_left:
        for ln in ("Breast.L", "Breast_L", "Breast.Left", "Left", "BreastL"):
            cand_names.add(ln)
            cand_names.add(ln.lower())
    if is_right:
        for rn in ("Breast.R", "Breast_R", "Breast.Right", "Right", "BreastR"):
            cand_names.add(rn)
            cand_names.add(rn.lower())

    was_mode = context.mode
    orig_active = context.view_layer.objects.active
    orig_active_name = orig_active.name if orig_active else None
    total_removed_bones = []

    for rig in rig_candidates:
        if not rig or rig.type != 'ARMATURE':
            continue

        ensure_object_visible(context, rig)
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

        try:
            context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            continue

        ebones = rig.data.edit_bones
        to_remove = set()

        # 2.1 이름 기반 매칭 (정확 일치, 대소문자 무시, 접두사/체인 일치)
        for eb in ebones:
            eb_l = eb.name.lower()
            for cn in cand_names:
                cn_l = cn.lower()
                if eb_l == cn_l or eb_l.startswith(f"{cn_l}.") or eb_l.startswith(f"{cn_l}_"):
                    to_remove.add(eb)
                    break

        # 2.2 이름으로 못 찾았고 슬롯 인덱스 idx가 유효한 경우 인덱스 매칭 fallback
        if not to_remove and 0 <= idx < len(ebones):
            eb_list = list(ebones)
            to_remove.add(eb_list[idx])

        # 본 삭제 실행
        for eb in list(to_remove):
            total_removed_bones.append(eb.name)
            ebones.remove(eb)

        remaining_count = len(ebones)
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        # 만약 리그의 모든 본이 삭제되었다면 빈 리그 오브젝트 제거
        if remaining_count == 0:
            if getattr(props, "anchor_obj", None) == rig:
                props.anchor_obj = None
            bpy.data.objects.remove(rig, do_unlink=True)

    # 원래 모드 복원
    if orig_active_name and orig_active_name in bpy.data.objects:
        try:
            o_act = bpy.data.objects[orig_active_name]
            if o_act.type == 'MESH':
                ensure_object_visible(context, o_act)
                context.view_layer.objects.active = o_act
                if was_mode == 'EDIT_MESH':
                    bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            pass

    return total_removed_bones, cand_names


class XXMI_OT_bone_item_remove(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_remove"
    bl_label = "물리 본 슬롯 삭제"
    bl_description = "선택된 물리 본 슬롯을 목록에서 삭제하고, 실제 블렌더 리그 상의 본과 버텍스 그룹도 함께 삭제합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and 0 <= props.bone_item_index < len(props.bone_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        idx = props.bone_item_index
        item = props.bone_items[idx]
        name = item.name

        # 1. 리그에서 해당 본 및 하위 체인 본 삭제
        removed_bones, cand_names = find_and_remove_bones_from_rigs(context, props, item, idx)

        # 2. 관련 메쉬 오브젝트의 버텍스 그룹 완전 제거
        vg_targets = {f"XXMI_{name}", name}
        for cn in cand_names:
            vg_targets.add(f"XXMI_{cn}")
            vg_targets.add(cn)

        cleared_vg_objs = 0
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                removed_any = False
                for vgn in vg_targets:
                    vg = obj.vertex_groups.get(vgn)
                    if vg:
                        obj.vertex_groups.remove(vg)
                        removed_any = True
                if removed_any:
                    cleared_vg_objs += 1

        # 3. 슬롯 목록에서 항목 삭제
        props.bone_items.remove(idx)
        props.bone_item_index = max(0, min(idx, len(props.bone_items) - 1))

        if removed_bones:
            bones_str = ", ".join(removed_bones)
            self.report({'INFO'}, f"물리 본 슬롯 '{name}' 및 리그 본 [{bones_str}] 삭제 완료.")
        else:
            self.report({'INFO'}, f"물리 본 슬롯 '{name}'을 삭제했습니다.")

        return {'FINISHED'}


class XXMI_OT_bone_item_assign(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_assign"
    bl_label = "선택 정점 재할당"
    bl_description = "에디트 모드에서 현재 선택된 정점들로 이 물리 본 슬롯의 정점 목록을 새로 교체(재할당)합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and 0 <= props.bone_item_index < len(props.bone_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items[props.bone_item_index]
        old_data = get_bone_data(item)

        target_objs = get_target_mesh_objects(context)
        if not target_objs:
            self.report({'ERROR'}, "선택된 메쉬 오브젝트가 없습니다.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        new_data = {}
        vg_target_name = f"XXMI_{item.name}"

        for obj in target_objs:
            mesh = obj.data
            n_verts = len(mesh.vertices)
            sel_mask = np.zeros(n_verts, dtype=bool)
            mesh.vertices.foreach_get("select", sel_mask)
            sel_indices = np.nonzero(sel_mask)[0].tolist()
            if sel_indices:
                new_data[obj.name] = sel_indices

        if not new_data:
            if was_edit:
                for obj in target_objs:
                    context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, "에디트 모드에서 슬롯에 재할당할 정점을 먼저 선택해주세요.")
            return {'CANCELLED'}

        # 기존 슬롯 정점들의 버텍스 그룹 초기화 후 새 선택 정점으로 교체
        affected_obj_names = set(old_data.keys()) | set(new_data.keys())
        for oname in affected_obj_names:
            obj = bpy.data.objects.get(oname)
            if not obj or obj.type != 'MESH':
                continue
            vg = obj.vertex_groups.get(vg_target_name)
            if vg:
                vg.remove(list(range(len(obj.data.vertices))))
            new_indices = new_data.get(oname, [])
            if new_indices:
                if not vg:
                    vg = obj.vertex_groups.new(name=vg_target_name)
                vg.add(new_indices, 0.05, 'REPLACE')

        if was_edit:
            for obj in target_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')

        set_bone_data(item, new_data)
        sync_slot_bone_to_rig(context, item)
        self.report({'INFO'}, f"본 '{item.name}'의 정점을 현재 선택된 {item.vert_count}개 정점으로 재할당 및 본 위치 갱신 완료!")
        return {'FINISHED'}


class XXMI_OT_bone_item_subtract_rigid_verts(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_subtract_rigid_verts"
    bl_label = "강체 정점 자동 제외"
    bl_description = "현재 물리 본 슬롯에서 등록된 강체 정점들을 자동 제외합니다 (표면과 맞닿아 있는 접합부 경계 버텍스는 자연스러운 연결을 위해 유지)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and (0 <= props.bone_item_index < len(props.bone_items)) and len(props.rigid_items) > 0

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items[props.bone_item_index]

        rigid_verts_by_obj = get_all_rigid_verts_by_obj(props)
        if not rigid_verts_by_obj:
            self.report({'WARNING'}, "등록된 강체 정점이 없습니다. 먼저 [2. 강체 설정]에서 강체를 등록해주세요.")
            return {'CANCELLED'}

        was_edit = (context.mode == 'EDIT_MESH')
        if was_edit:
            ensure_all_objects_in_object_mode(context)

        subtracted, kept = subtract_rigid_from_bone_slot_data(item, rigid_verts_by_obj, keep_boundary=True)

        # 강체 제외 후 순수 물리 영역에 맞춰 본 위치 갱신
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
    bl_label = "모든 물리 본에서 강체 정점 자동 제외"
    bl_description = "모든 물리 본 슬롯에서 등록된 강체 정점들을 일괄 제외합니다 (표면 접합부 경계 정점 보존)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and len(props.bone_items) > 0 and len(props.rigid_items) > 0

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

        self.report({'INFO'}, f"전체 물리 본에서 강체 정점 {total_sub}개 제외 완료 (경계 정점 {total_kept}개 유지).")
        return {'FINISHED'}


class XXMI_OT_bone_item_select_verts(bpy.types.Operator):
    bl_idname = "object.xxmi_bone_item_select_verts"
    bl_label = "정점 확인"
    bl_description = "이 본에 할당된 정점들을 에디트 모드에서 선택하여 확인합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and 0 <= props.bone_item_index < len(props.bone_items)

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        item = props.bone_items[props.bone_item_index]
        data = get_bone_data(item)

        # 1. data_json이 비어있거나 불완전한 경우, 씬 내 메쉬의 버텍스 그룹으로부터 자동 복구 시도
        if not data:
            cand_vg_names = [f"XXMI_{item.name}"]
            if "왼" in item.name or ".L" in item.name.upper() or "_L" in item.name.upper():
                cand_vg_names.extend(["XXMI_Breast_L", "XXMI_Breast.L"])
            elif "오" in item.name or ".R" in item.name.upper() or "_R" in item.name.upper():
                cand_vg_names.extend(["XXMI_Breast_R", "XXMI_Breast.R"])

            recovered = {}
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    vg = None
                    for vgn in cand_vg_names:
                        vg = obj.vertex_groups.get(vgn)
                        if vg:
                            break
                    if vg:
                        vis = []
                        for vi in range(len(obj.data.vertices)):
                            try:
                                if vg.weight(vi) > 0.0001:
                                    vis.append(vi)
                            except Exception:
                                pass
                        if vis:
                            recovered[obj.name] = vis
            if recovered:
                data = recovered
                set_bone_data(item, data)

        if not data:
            self.report({'WARNING'}, f"본 '{item.name}'에 등록된 정점이 없습니다.")
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        first_obj = None

        # 이름 불일치 대비 메쉬 매핑 준비
        resolved_mesh_map = {}
        for obj_name, v_indices in data.items():
            obj = bpy.data.objects.get(obj_name)
            if not obj or obj.type != 'MESH':
                # 오브젝트 이름이 바뀌었거나 다른 씬일 경우 대체 탐색
                for o in bpy.data.objects:
                    if o.type == 'MESH' and (o.vertex_groups.get(f"XXMI_{item.name}") or o.vertex_groups.get("XXMI_Breast_L") or o.vertex_groups.get("XXMI_Breast_R")):
                        obj = o
                        break
                if not obj and context.active_object and context.active_object.type == 'MESH':
                    obj = context.active_object

            if obj and obj.type == 'MESH':
                resolved_mesh_map[obj] = v_indices
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
        for obj, v_indices in resolved_mesh_map.items():
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
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and 0 <= props.bone_item_index < len(props.bone_items)

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
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and len(props.bone_items) > 1 and (0 <= props.bone_item_index < len(props.bone_items))

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        curr = props.bone_items[props.bone_item_index]

        count = 0
        for i, item in enumerate(props.bone_items):
            if i != props.bone_item_index:
                item.radius_scale = curr.radius_scale
                item.peak_weight = curr.peak_weight
                item.falloff_power = curr.falloff_power
                item.smooth_iterations = curr.smooth_iterations
                item.lock_registered_verts = curr.lock_registered_verts
                count += 1

        self.report({'INFO'}, f"본 '{curr.name}'의 가중치 설정을 {count}개 본에 복사했습니다.")
        return {'FINISHED'}


class XXMI_OT_reposition_physics_bones(bpy.types.Operator):
    bl_idname = "object.xxmi_generate_breast_rig"  # 하위 호환 유지
    bl_label = "모든 물리 본 위치/크기 일괄 재배치"
    bl_description = "등록된 모든 물리 본 슬롯의 정점 3D 기하학적 형태 중심을 정밀 분석하여 물리 본(XXMI_Physics_Rig)을 자동 배치합니다"
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
            self.report({'ERROR'}, "등록된 물리 본 슬롯이 없습니다. 먼저 [+] 버튼을 눌러 슬롯을 생성하세요.")
            return {'CANCELLED'}

        synced_count = reposition_all_slot_bones(context, props)
        if synced_count == 0:
            self.report({'WARNING'}, "본 슬롯에 할당된 정점이 부족하여 본을 배치하지 못했습니다. 에디트 모드에서 정점을 선택 후 [선택 정점 할당]을 진행하세요.")
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"총 {synced_count}개 물리 본 위치 및 크기 배치 완료! 뷰포트에서 본의 위치를 확인하고 필요시 에디트 모드에서 미세 조정한 뒤 3패널에서 가중치를 계산하세요."
        )
        return {'FINISHED'}


# 별칭 클래스
XXMI_OT_generate_breast_rig = XXMI_OT_reposition_physics_bones


class XXMI_OT_rebake_physics_weights(bpy.types.Operator):
    bl_idname = "object.xxmi_rebake_breast_weights"  # 하위 호환 유지
    bl_label = "본 & 강체 가중치 자동 계산 & 블렌딩"
    bl_description = "물리 본들을 기준으로 거리 감쇄 가중치를 계산하고 강체를 포함하여 자연스럽게 블렌딩합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        if not props:
            return False
        rig = getattr(props, "anchor_obj", None) or bpy.data.objects.get(PHYSICS_RIG_NAME)
        return rig is not None and rig.type == 'ARMATURE'

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        rig = getattr(props, "anchor_obj", None) or bpy.data.objects.get(PHYSICS_RIG_NAME)
        if not rig:
            self.report({'ERROR'}, "물리 본을 찾을 수 없습니다. 1패널에서 슬롯 생성 및 [모든 물리 본 위치/크기 일괄 재배치]를 먼저 진행해주세요.")
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

        self.report({'INFO'}, f"물리 가중치 계산 완료: {msg}")
        return {'FINISHED'}


# 별칭 클래스
XXMI_OT_rebake_breast_weights = XXMI_OT_rebake_physics_weights


class XXMI_OT_clear_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_clear_breast_physics"  # 하위 호환 유지
    bl_label = "물리 본 & 가중치 전체 초기화"
    bl_description = "생성된 물리 본 리그와 모든 가중치 설정을 지우고 초기화합니다"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, "xxmi_custom_physics_props", None)
        return props is not None and (props.anchor_obj is not None or bpy.data.objects.get(PHYSICS_RIG_NAME) is not None or len(props.bone_items) > 0)

    def execute(self, context):
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        from ..simulation.engine import stop_physics_preview
        props = context.scene.xxmi_custom_physics_props
        if props.is_preview_running:
            stop_physics_preview(context)

        for r_name in (PHYSICS_RIG_NAME, "XXMI_Breast_Rig"):
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

        self.report({'INFO'}, f"물리 설정 초기화 완료 ({cleared_objs}개 메쉬 오브젝트 가중치 제거됨).")
        return {'FINISHED'}


# 별칭 클래스
XXMI_OT_clear_breast_physics = XXMI_OT_clear_physics


classes = (
    XXMI_OT_bone_item_add,
    XXMI_OT_bone_item_add_split_lr,
    XXMI_OT_bone_item_remove,
    XXMI_OT_bone_item_assign,
    XXMI_OT_bone_item_select_verts,
    XXMI_OT_bone_item_remove_verts,
    XXMI_OT_bone_item_subtract_rigid_verts,
    XXMI_OT_subtract_rigid_from_all_bones,
    XXMI_OT_copy_bone_weights_to_all,
    XXMI_OT_reposition_physics_bones,
    XXMI_OT_rebake_physics_weights,
    XXMI_OT_clear_physics,
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
