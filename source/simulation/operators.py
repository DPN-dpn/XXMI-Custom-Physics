import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import (
    PREVIEW_SK_NAME,
    PREVIEW_BASIS_TAG,
    ensure_object_visible,
    ensure_all_objects_in_object_mode,
    get_local_axes,
)
from .engine import (
    _PREVIEW_CACHE,
    preview_frame_handler,
    stop_physics_preview,
)

# 하위 호환용 별칭
_preview_frame_handler = preview_frame_handler


class XXMI_OT_preview_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_preview_physics"
    bl_label = "뷰포트 실시간 물리 프리뷰"
    bl_description = "뷰포트에서 물리 본의 움직임을 실시간으로 미리 확인합니다"

    @classmethod
    def poll(cls, context):
        return any(
            o.type == 'MESH' and any(vg.name.startswith("XXMI_") for vg in o.vertex_groups)
            for o in bpy.data.objects
        )

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props

        if props.is_preview_running:
            stop_physics_preview(context)
            self.report({'INFO'}, "물리 프리뷰가 중지되었으며 메쉬들이 원상 복구되었습니다.")
            return {'FINISHED'}

        ensure_all_objects_in_object_mode(context)

        # 1. 활성 본 슬롯 목록 추출
        active_bones = []
        if props and hasattr(props, "bone_items") and len(props.bone_items) > 0:
            active_bones = [it.name for it in props.bone_items if it.name.strip()]

        if not active_bones:
            # 씬 내 메쉬의 버텍스 그룹에서 탐색
            bone_set = set()
            for o in bpy.data.objects:
                if o.type == 'MESH':
                    for vg in o.vertex_groups:
                        if vg.name.startswith("XXMI_") and vg.name not in ("XXMI_Role_Rigid", "XXMI_Role_Static", "XXMI_Physics"):
                            bone_set.add(vg.name[5:])
            active_bones = sorted(list(bone_set))

        if not active_bones:
            self.report({'ERROR'}, "등록된 모핑 본 슬롯이나 가중치 버텍스 그룹이 없습니다. 먼저 1패널에서 슬롯을 생성하세요.")
            return {'CANCELLED'}

        # 2. 관련 메쉬 오브젝트 탐색
        active_vg_names = [f"XXMI_{bn}" for bn in active_bones]
        legacy_vg_names = ["XXMI_Breast_L", "XXMI_Breast_R"]
        all_candidate_vgs = set(active_vg_names + legacy_vg_names + ["XXMI_Role_Rigid"])

        mesh_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and any(o.vertex_groups.get(vgn) for vgn in all_candidate_vgs)
        ]

        if not mesh_objs:
            self.report({'ERROR'}, "물리 가중치가 적용된 메쉬 오브젝트를 찾을 수 없습니다.")
            return {'CANCELLED'}

        objects_data = []

        for obj in mesh_objs:
            ensure_object_visible(context, obj)
            mesh = obj.data
            n_verts = len(mesh.vertices)

            # 본별 가중치 배열 수집
            weights_by_bone = {}
            has_any_weights = False

            for bn in active_bones:
                vg = obj.vertex_groups.get(f"XXMI_{bn}")
                # 레거시 매핑 호환
                if not vg and bn.lower() in ("breast.l", "l"):
                    vg = obj.vertex_groups.get("XXMI_Breast_L")
                elif not vg and bn.lower() in ("breast.r", "r"):
                    vg = obj.vertex_groups.get("XXMI_Breast_R")

                w_arr = [0.0] * n_verts
                if vg:
                    vg_idx = vg.index
                    for v in mesh.vertices:
                        for g in v.groups:
                            if g.group == vg_idx and g.weight > 1e-4:
                                w_arr[v.index] = g.weight
                                has_any_weights = True
                weights_by_bone[bn] = w_arr

            vg_rigid = obj.vertex_groups.get("XXMI_Role_Rigid")
            rigid_flags = [False] * n_verts
            if vg_rigid:
                vr_idx = vg_rigid.index
                for v in mesh.vertices:
                    for g in v.groups:
                        if g.group == vr_idx and g.weight > 0.5:
                            rigid_flags[v.index] = True
                            has_any_weights = True

            if not has_any_weights:
                continue

            up_local, right_local, fwd_local = get_local_axes(obj)

            if not mesh.shape_keys:
                obj.shape_key_add(name="Basis", from_mix=False)
                obj[PREVIEW_BASIS_TAG] = True
            elif PREVIEW_SK_NAME in mesh.shape_keys.key_blocks:
                obj.shape_key_remove(mesh.shape_keys.key_blocks[PREVIEW_SK_NAME])

            sk = obj.shape_key_add(name=PREVIEW_SK_NAME, from_mix=False)
            sk.value = 1.0

            active_indices = [
                i for i in range(n_verts)
                if any(weights_by_bone[bn][i] > 1e-4 for bn in active_bones) or rigid_flags[i]
            ]

            objects_data.append({
                "obj_name": obj.name,
                "active_indices": active_indices,
                "weights_by_bone": weights_by_bone,
                "rigid_flags": rigid_flags,
                "base_coords": [v.co.copy() for v in mesh.vertices],
                "up_vec": up_local,
                "right_vec": right_local,
                "fwd_vec": fwd_local,
            })

        if not objects_data:
            self.report({'ERROR'}, "물리 가중치가 계산되지 않았습니다. 먼저 3패널에서 [가중치 자동 계산 & 블렌딩]을 진행해주세요.")
            return {'CANCELLED'}

        global_peak_w = 0.0
        for odata in objects_data:
            for bn in active_bones:
                w_arr = odata["weights_by_bone"].get(bn, [])
                if w_arr:
                    m = max(w_arr)
                    if m > global_peak_w:
                        global_peak_w = m

        _PREVIEW_CACHE["objects_data"] = objects_data
        _PREVIEW_CACHE["active_bones"] = active_bones
        _PREVIEW_CACHE["norm_factor"] = (1.0 / global_peak_w) if global_peak_w > 0.05 else 1.0
        _PREVIEW_CACHE["ode_state"] = {
            "bones": {},
            "prev_t": None,
        }
        props.is_preview_running = True

        if preview_frame_handler not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(preview_frame_handler)

        preview_frame_handler(context.scene)

        try:
            if context.screen and not context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
        except Exception:
            pass

        self.report({'INFO'}, f"물리 프리뷰 시작 ({len(objects_data)}개 메쉬 오브젝트, 본 {len(active_bones)}개)")
        return {'FINISHED'}


class XXMI_OT_copy_physics_params_to_all(bpy.types.Operator):
    bl_idname = "object.xxmi_copy_physics_params_to_all"
    bl_label = "모든 본에 현재 설정 일괄 복사"
    bl_description = "현재 선택된 본의 가중치 및 물리 파라미터를 모든 본에 동일하게 복사합니다"
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
                # 가중치 설정 복사
                item.peak_weight = curr.peak_weight
                item.falloff_power = curr.falloff_power
                item.smooth_iterations = curr.smooth_iterations
                item.lock_registered_verts = curr.lock_registered_verts
                # 물리 파라미터 복사
                item.strength = curr.strength
                item.speed = curr.speed
                item.damping = curr.damping
                item.inertia = curr.inertia
                item.idle_motion = curr.idle_motion
                count += 1

        self.report({'INFO'}, f"본 '{curr.name}'의 가중치 및 물리 파라미터를 {count}개 본에 복사했습니다.")
        return {'FINISHED'}


class XXMI_OT_reset_physics_params(bpy.types.Operator):
    bl_idname = "object.xxmi_reset_physics_params"
    bl_label = "기본값으로 초기화"
    bl_description = "가중치 및 물리 파라미터를 권장 기본값으로 되돌립니다"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        props.peak_weight = 0.40
        props.falloff_power = 1.0
        props.smooth_iterations = 2
        props.lock_registered_verts = True

        props.strength = 1.0
        props.speed = 1.1
        props.damping = 0.85
        props.inertia = 1.2
        props.idle_motion = 0.4

        if 0 <= props.bone_item_index < len(props.bone_items):
            curr = props.bone_items[props.bone_item_index]
            curr.peak_weight = 0.40
            curr.falloff_power = 1.0
            curr.smooth_iterations = 2
            curr.lock_registered_verts = True

            curr.strength = 1.0
            curr.speed = 1.1
            curr.damping = 0.85
            curr.inertia = 1.2
            curr.idle_motion = 0.4
            self.report({'INFO'}, f"본 '{curr.name}'의 가중치 및 물리 파라미터가 기본값으로 초기화되었습니다.")
        else:
            self.report({'INFO'}, "가중치 및 물리 파라미터가 기본값으로 초기화되었습니다.")
        return {'FINISHED'}


classes = (
    XXMI_OT_preview_physics,
    XXMI_OT_copy_physics_params_to_all,
    XXMI_OT_reset_physics_params,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    if preview_frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(preview_frame_handler)
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
