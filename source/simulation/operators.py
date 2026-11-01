import bpy
import math
from mathutils import Vector, Matrix
from ..utils import (
    PREVIEW_SK_NAME,
    PREVIEW_BASIS_TAG,
    ensure_object_visible,
    get_local_axes,
    axis_angle_matrix,
)

# 다중 오브젝트 프리뷰 런타임 캐시
_PREVIEW_CACHE = {
    "objects_data": [],  # list of dicts per animated mesh object
}


def _evaluate_breast_transform(t, side_sign, props, up_vec, right_vec, fwd_vec):
    """
    시간 t에서 한쪽 가슴의 탄성 변위 벡터(Vector3)를 계산합니다. (인게임 HLSL과 100% 동일한 선형 탄성)
    side_sign: 1.0 (Left), -1.0 (Right)
    """
    amp = props.strength
    spd = max(0.2, props.speed)
    damp = max(0.1, props.damping)
    inertia = props.inertia
    idle = props.idle_motion

    # 이동 / 착지 / 턴 리듬 파형 (DC 오프셋 없는 순수 0-중심 조화 진동)
    cycle = (t * spd * 1.8) % 3.0
    if cycle < 1.6:
        # 달리기 중 발걸음 충격 및 상하/전후 탄성
        step_phase = cycle / 1.6 * math.pi * 2.0
        s_z = math.sin(step_phase * 2.0) * 0.8 * inertia
        s_y = math.sin(step_phase) * 0.6 * inertia
        s_x = math.cos(step_phase) * 0.4 * inertia * side_sign
    else:
        # 정지 시 여운 감쇠 진동 (잔여 탄성)
        dt_stop = cycle - 1.6
        decay = math.exp(-dt_stop * 4.0 * damp)
        s_z = math.sin(dt_stop * 14.0 * spd) * 0.9 * decay * inertia
        s_y = math.cos(dt_stop * 12.0 * spd) * 0.7 * decay * inertia
        s_x = math.sin(dt_stop * 10.0 * spd) * 0.4 * decay * inertia * side_sign

    # 미세 유휴 호흡 모션
    s_y += math.sin(t * 2.5) * 0.15 * idle
    s_z += math.sin(t * 2.5 + 0.5) * 0.10 * idle

    # 3차원 위치 변위 (인게임 스케일 2~3cm 선형 변위)
    trans = (
        up_vec * (s_z * 0.012) +
        fwd_vec * (s_y * 0.010) +
        right_vec * (s_x * 0.007)
    ) * amp

    return trans


def _preview_frame_handler(scene):
    props = getattr(scene, "xxmi_custom_physics_props", None)
    if not props or not props.is_preview_running:
        return

    objects_data = _PREVIEW_CACHE.get("objects_data", [])
    if not objects_data:
        return

    fps = scene.render.fps / max(scene.render.fps_base, 1e-4)
    t = scene.frame_current / max(fps, 24.0)

    for odata in objects_data:
        obj_name = odata["obj_name"]
        if obj_name not in bpy.data.objects:
            continue

        obj = bpy.data.objects[obj_name]
        mesh = obj.data
        if not mesh.shape_keys or PREVIEW_SK_NAME not in mesh.shape_keys.key_blocks:
            continue

        sk = mesh.shape_keys.key_blocks[PREVIEW_SK_NAME]
        base_coords = odata["base_coords"]
        active_indices = odata["active_indices"]
        w_l = odata["weights_l"]
        w_r = odata["weights_r"]
        up_vec = odata["up_vec"]
        right_vec = odata["right_vec"]
        fwd_vec = odata["fwd_vec"]

        if len(sk.data) != len(base_coords):
            continue

        trans_l = _evaluate_breast_transform(t, 1.0, props, up_vec, right_vec, fwd_vec)
        trans_r = _evaluate_breast_transform(t, -1.0, props, up_vec, right_vec, fwd_vec)

        # 활성 가슴 정점(~3,000개)만 초고속 연산 (렉 및 속도 저하 완벽 해결)
        for i in active_indices:
            base_co = base_coords[i]
            wl = w_l[i]
            wr = w_r[i]
            total_disp = trans_l * wl + trans_r * wr
            sk.data[i].co = base_co + total_disp


def stop_physics_preview(context):
    """실행 중인 뷰포트 프리뷰를 중지하고 모든 대상 오브젝트의 쉐이프키를 완벽히 정리합니다."""
    if _preview_frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_preview_frame_handler)

    try:
        if context.screen and context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()
    except Exception:
        pass

    objects_data = _PREVIEW_CACHE.get("objects_data", [])
    for odata in objects_data:
        obj_name = odata.get("obj_name")
        if obj_name and obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            if obj.type == 'MESH' and obj.data.shape_keys:
                kb = obj.data.shape_keys.key_blocks
                if PREVIEW_SK_NAME in kb:
                    obj.shape_key_remove(kb[PREVIEW_SK_NAME])
                if PREVIEW_BASIS_TAG in obj:
                    if len(kb) == 1 and kb[0].name == "Basis":
                        obj.shape_key_remove(kb[0])
                    del obj[PREVIEW_BASIS_TAG]

    _PREVIEW_CACHE["objects_data"] = []

    if hasattr(context, "scene") and hasattr(context.scene, "xxmi_custom_physics_props"):
        context.scene.xxmi_custom_physics_props.is_preview_running = False


class XXMI_OT_preview_physics(bpy.types.Operator):
    bl_idname = "object.xxmi_preview_physics"
    bl_label = "뷰포트 실시간 바운스 프리뷰"
    bl_description = "가슴 물리 웨이트가 등록된 모든 메쉬 오브젝트(Body, 의상 등)의 바운스 모핑을 실시간 애니메이션으로 동시 미리보기합니다"

    @classmethod
    def poll(cls, context):
        # 씬 내에 가슴 웨이트 그룹이 존재하는 메쉬가 1개 이상 있으면 활성화
        return any(
            o.type == 'MESH' and (o.vertex_groups.get("XXMI_Breast_L") or o.vertex_groups.get("XXMI_Breast_R"))
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

        # 가슴 웨이트가 있는 모든 메쉬 수집
        breast_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and (o.vertex_groups.get("XXMI_Breast_L") or o.vertex_groups.get("XXMI_Breast_R"))
        ]

        if not breast_objs:
            self.report({'ERROR'}, "1단계에서 가슴 선택 영역을 먼저 등록해주세요 (XXMI_Breast_L/R 그룹 없음).")
            return {'CANCELLED'}

        # 공통 앵커(XXMI_Breast_Rig) 월드 좌표 취득
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

            if not has_weights:
                continue

            inv_mw = obj.matrix_world.inverted()
            local_anchor_l = inv_mw @ world_anchor_l if world_anchor_l.length_squared > 1e-6 else Vector((0.0, 0.0, 0.0))
            local_anchor_r = inv_mw @ world_anchor_r if world_anchor_r.length_squared > 1e-6 else Vector((0.0, 0.0, 0.0))

            up_local, right_local, fwd_local = get_local_axes(obj)

            # 쉐이프키 준비
            if not mesh.shape_keys:
                obj.shape_key_add(name="Basis", from_mix=False)
                obj[PREVIEW_BASIS_TAG] = True
            elif PREVIEW_SK_NAME in mesh.shape_keys.key_blocks:
                obj.shape_key_remove(mesh.shape_keys.key_blocks[PREVIEW_SK_NAME])

            sk = obj.shape_key_add(name=PREVIEW_SK_NAME, from_mix=False)
            sk.value = 1.0

            active_indices = [i for i in range(n_verts) if w_l[i] > 1e-5 or w_r[i] > 1e-5]

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
            self.report({'ERROR'}, "유효한 가슴 가중치를 가진 메쉬가 없습니다.")
            return {'CANCELLED'}

        _PREVIEW_CACHE["objects_data"] = objects_data
        props.is_preview_running = True

        if _preview_frame_handler not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(_preview_frame_handler)

        _preview_frame_handler(context.scene)

        try:
            if context.screen and not context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
        except Exception:
            pass

        obj_list_str = ", ".join(d["obj_name"] for d in objects_data)
        self.report({'INFO'}, f"다중 오브젝트({len(objects_data)}개: {obj_list_str}) 가슴 물리 동시 프리뷰 재생 시작!")
        return {'FINISHED'}


class XXMI_OT_reset_physics_params(bpy.types.Operator):
    bl_idname = "object.xxmi_reset_physics_params"
    bl_label = "파라미터 기본값으로 초기화"
    bl_description = "가슴 물리 및 관성 파라미터(강도, 속도, 감쇠, 관성, 호흡 모션)를 권장 기본값으로 되돌립니다"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        props.strength = 1.0
        props.speed = 1.1
        props.damping = 0.85
        props.inertia = 1.2
        props.idle_motion = 0.8
        props.falloff_power = 1.6
        self.report({'INFO'}, "가슴 물리 파라미터가 권장 기본값으로 초기화되었습니다.")
        return {'FINISHED'}


classes = (
    XXMI_OT_preview_physics,
    XXMI_OT_reset_physics_params,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    if _preview_frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(_preview_frame_handler)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
