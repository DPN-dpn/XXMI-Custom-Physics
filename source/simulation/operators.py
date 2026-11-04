import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import (
    PREVIEW_SK_NAME,
    PREVIEW_BASIS_TAG,
    ensure_object_visible,
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
    bl_description = "뷰포트에서 가슴 물리 움직임을 실시간으로 미리 확인합니다"

    @classmethod
    def poll(cls, context):
        return any(
            o.type == 'MESH' and (
                o.vertex_groups.get("XXMI_Breast_L") or
                o.vertex_groups.get("XXMI_Breast_R") or
                o.vertex_groups.get("XXMI_Role_Rigid")
            )
            for o in bpy.data.objects
        )

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props

        if props.is_preview_running:
            stop_physics_preview(context)
            self.report({'INFO'}, "가슴 물리 프리뷰가 중지되었으며 메쉬들이 원상 복구되었습니다.")
            return {'FINISHED'}

        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        breast_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and (
                o.vertex_groups.get("XXMI_Breast_L") or
                o.vertex_groups.get("XXMI_Breast_R") or
                o.vertex_groups.get("XXMI_Role_Rigid")
            )
        ]

        if not breast_objs:
            self.report({'ERROR'}, "등록된 가슴 정점이 없습니다. 먼저 [1. 가슴 영역 선택 등록]을 진행해주세요.")
            return {'CANCELLED'}

        rig = props.anchor_obj or bpy.data.objects.get("XXMI_Breast_Rig")
        world_anchor_l = Vector((0.0, 0.0, 0.0))
        world_anchor_r = Vector((0.0, 0.0, 0.0))
        if rig and rig.type == 'ARMATURE':
            b_l = rig.data.bones.get("Breast.L")
            b_r = rig.data.bones.get("Breast.R")
            if b_l:
                world_anchor_l = rig.matrix_world @ b_l.head_local
            if b_r:
                world_anchor_r = rig.matrix_world @ b_r.head_local

        objects_data = []

        for obj in breast_objs:
            ensure_object_visible(context, obj)
            mesh = obj.data
            n_verts = len(mesh.vertices)
            vg_l_idx = obj.vertex_groups.find("XXMI_Breast_L")
            vg_r_idx = obj.vertex_groups.find("XXMI_Breast_R")
            vg_rigid_idx = obj.vertex_groups.find("XXMI_Role_Rigid")

            w_l = [0.0] * n_verts
            w_r = [0.0] * n_verts
            rigid_flags = [False] * n_verts

            has_weights = False
            for v in mesh.vertices:
                for g in v.groups:
                    if g.group == vg_l_idx:
                        w_l[v.index] = g.weight
                        if g.weight > 0:
                            has_weights = True
                    elif g.group == vg_r_idx:
                        w_r[v.index] = g.weight
                        if g.weight > 0:
                            has_weights = True
                    elif g.group == vg_rigid_idx and g.weight > 0.5:
                        rigid_flags[v.index] = True
                        has_weights = True

            if not has_weights:
                continue

            inv_mw = obj.matrix_world.inverted()
            local_anchor_l = inv_mw @ world_anchor_l if world_anchor_l.length_squared > 1e-6 else Vector((0.0, 0.0, 0.0))
            local_anchor_r = inv_mw @ world_anchor_r if world_anchor_r.length_squared > 1e-6 else Vector((0.0, 0.0, 0.0))

            up_local, right_local, fwd_local = get_local_axes(obj)

            if not mesh.shape_keys:
                obj.shape_key_add(name="Basis", from_mix=False)
                obj[PREVIEW_BASIS_TAG] = True
            elif PREVIEW_SK_NAME in mesh.shape_keys.key_blocks:
                obj.shape_key_remove(mesh.shape_keys.key_blocks[PREVIEW_SK_NAME])

            sk = obj.shape_key_add(name=PREVIEW_SK_NAME, from_mix=False)
            sk.value = 1.0

            active_indices = [i for i in range(n_verts) if w_l[i] > 1e-5 or w_r[i] > 1e-5 or rigid_flags[i]]

            objects_data.append({
                "obj_name": obj.name,
                "anchor_l": local_anchor_l,
                "anchor_r": local_anchor_r,
                "active_indices": active_indices,
                "weights_l": w_l,
                "weights_r": w_r,
                "rigid_flags": rigid_flags,
                "base_coords": [v.co.copy() for v in mesh.vertices],
                "up_vec": up_local,
                "right_vec": right_local,
                "fwd_vec": fwd_local,
            })

        if not objects_data:
            self.report({'ERROR'}, "가슴 가중치가 계산되지 않았습니다. 먼저 [3. 본 기준 가중치(웨이트) 자동 계산]을 진행해주세요.")
            return {'CANCELLED'}

        all_pts_preview = []
        for o_idx, odata in enumerate(objects_data):
            obj = bpy.data.objects.get(odata["obj_name"])
            if not obj:
                continue
            mw = obj.matrix_world
            base_coords = odata["base_coords"]
            rigid_flags = odata["rigid_flags"]
            for vi in odata["active_indices"]:
                if rigid_flags[vi]:
                    continue
                w_co = mw @ base_coords[vi]
                all_pts_preview.append((w_co, o_idx, vi))

        if len(all_pts_preview) > 1:
            kd_prev = KDTree(len(all_pts_preview))
            for k, (w_co, _, _) in enumerate(all_pts_preview):
                kd_prev.insert(w_co, k)
            kd_prev.balance()

            for w_co, o_idx, vi in all_pts_preview:
                hits = kd_prev.find_range(w_co, 0.0012)
                if len(hits) > 1:
                    max_wl = max(objects_data[all_pts_preview[h_k][1]]["weights_l"][all_pts_preview[h_k][2]] for _, h_k, _ in hits)
                    max_wr = max(objects_data[all_pts_preview[h_k][1]]["weights_r"][all_pts_preview[h_k][2]] for _, h_k, _ in hits)
                    objects_data[o_idx]["weights_l"][vi] = max_wl
                    objects_data[o_idx]["weights_r"][vi] = max_wr

        global_peak_w = 0.0
        for odata in objects_data:
            max_l = max(odata["weights_l"]) if odata["weights_l"] else 0.0
            max_r = max(odata["weights_r"]) if odata["weights_r"] else 0.0
            global_peak_w = max(global_peak_w, max_l, max_r)

        _PREVIEW_CACHE["objects_data"] = objects_data
        _PREVIEW_CACHE["norm_factor"] = (1.0 / global_peak_w) if global_peak_w > 0.05 else 1.0
        _PREVIEW_CACHE["ode_state"] = {
            "disp_l": Vector((0.0, 0.0, 0.0)),
            "disp_r": Vector((0.0, 0.0, 0.0)),
            "vel_l": Vector((0.0, 0.0, 0.0)),
            "vel_r": Vector((0.0, 0.0, 0.0)),
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

        self.report({'INFO'}, f"물리 프리뷰 시작 ({len(objects_data)}개 메쉬 오브젝트)")
        return {'FINISHED'}


class XXMI_OT_copy_physics_params_to_all(bpy.types.Operator):
    bl_idname = "object.xxmi_copy_physics_params_to_all"
    bl_label = "모든 본에 현재 파라미터 일괄 복사"
    bl_description = "현재 선택된 본의 물리 파라미터(흔들림, 탄력, 감쇠, 관성 등)를 모든 본에 동일하게 복사합니다"
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
                item.strength = curr.strength
                item.speed = curr.speed
                item.damping = curr.damping
                item.inertia = curr.inertia
                item.idle_motion = curr.idle_motion
                count += 1

        self.report({'INFO'}, f"본 '{curr.name}'의 물리 파라미터를 {count}개 본에 복사했습니다.")
        return {'FINISHED'}


class XXMI_OT_reset_physics_params(bpy.types.Operator):
    bl_idname = "object.xxmi_reset_physics_params"
    bl_label = "기본값으로 초기화"
    bl_description = "물리 파라미터를 권장 기본값으로 되돌립니다"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        props.strength = 1.0
        props.speed = 1.1
        props.damping = 0.85
        props.inertia = 1.2
        props.idle_motion = 0.4
        props.falloff_power = 1.0

        if 0 <= props.bone_item_index < len(props.bone_items):
            curr = props.bone_items[props.bone_item_index]
            curr.strength = 1.0
            curr.speed = 1.1
            curr.damping = 0.85
            curr.inertia = 1.2
            curr.idle_motion = 0.4
            self.report({'INFO'}, f"본 '{curr.name}'의 물리 파라미터가 기본값으로 초기화되었습니다.")
        else:
            self.report({'INFO'}, "물리 파라미터가 기본값으로 초기화되었습니다.")
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
