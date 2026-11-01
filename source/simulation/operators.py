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

# 다중 오브젝트 프리뷰 런타임 캐시 (인게임 HLSL ODE 상태와 1:1 동기화)
_PREVIEW_CACHE = {
    "objects_data": [],  # list of dicts per animated mesh object
    "norm_factor": 1.0,
    "ode_state": {
        "disp_l": Vector((0.0, 0.0, 0.0)),
        "disp_r": Vector((0.0, 0.0, 0.0)),
        "vel_l": Vector((0.0, 0.0, 0.0)),
        "vel_r": Vector((0.0, 0.0, 0.0)),
        "prev_t": None,
    },
}


def _step_preview_ode(t, props):
    """
    인게임 CustomPhysicsMotion.hlsl과 100% 동일한 4-Substep Symplectic Euler ODE 적분을 수행하여
    좌/우 가슴의 물리 변위(disp_l, disp_r)를 반환합니다.
    """
    state = _PREVIEW_CACHE["ode_state"]
    prev_t = state["prev_t"]

    if prev_t is None or t < prev_t or (t - prev_t) > 0.25:
        dt = 1.0 / 60.0
        if prev_t is None:
            state["disp_l"] = Vector((0.0, 0.0, 0.0))
            state["disp_r"] = Vector((0.0, 0.0, 0.0))
            state["vel_l"] = Vector((0.0, 0.0, 0.0))
            state["vel_r"] = Vector((0.0, 0.0, 0.0))
    else:
        raw_dt = t - prev_t
        if abs(raw_dt) <= 0.0005:
            return state["disp_l"], state["disp_r"]
        dt = max(0.002, min(0.04, raw_dt))

    state["prev_t"] = t

    speed = max(0.2, float(props.speed))
    damping = max(0.1, float(props.damping))
    inertia = max(0.0, float(props.inertia))
    idle_motion = max(0.0, float(props.idle_motion))

    omega_0 = 12.0 * speed
    stiff_k = omega_0 * omega_0
    damp_c = 2.0 * omega_0 * (damping * 0.50)

    # 3.0초 주기의 인게임 표준 몸통 모션 시뮬레이션:
    # 0.0 ~ 1.2초: 보행/대시 이동 발생 (관성 가속도 입력)
    # 1.2 ~ 3.0초: 정지 후 잔여 탄성 감쇠(Damping) 및 호흡 미동(Idle) 확인 구간
    cycle = t % 3.0
    if cycle < 1.2:
        ph = (cycle / 1.2) * math.pi * 2.0
        raw_vel = Vector((
            math.cos(ph) * 0.14,
            math.sin(ph) * 0.15,
            math.sin(ph * 2.0) * 0.20,
        ))
    else:
        raw_vel = Vector((0.0, 0.0, 0.0))

    anim_vel = raw_vel / (1.0 + raw_vel.length * 0.8)
    ext_force = -anim_vel * (12.0 * inertia)

    ext_force_l = ext_force.copy()
    ext_force_r = Vector((-ext_force.x * 0.9, ext_force.y, ext_force.z))

    if idle_motion > 0.001:
        breath_phase = t * 2.5
        ext_force_l.z += math.sin(breath_phase) * 1.5 * idle_motion
        ext_force_l.y += math.cos(breath_phase) * 1.1 * idle_motion
        ext_force_l.x += math.sin(breath_phase * 0.5) * 0.6 * idle_motion

        ext_force_r.z += math.sin(breath_phase + 0.35) * 1.5 * idle_motion
        ext_force_r.y += math.cos(breath_phase + 0.35) * 1.1 * idle_motion
        ext_force_r.x += -math.sin(breath_phase * 0.5 + 0.35) * 0.6 * idle_motion

    stiff_l = stiff_k * 1.04
    stiff_r = stiff_k * 0.96
    damp_l = damp_c * 0.98
    damp_r = damp_c * 1.02

    disp_l = state["disp_l"]
    disp_r = state["disp_r"]
    vel_l = state["vel_l"]
    vel_r = state["vel_r"]

    sub_dt = dt * 0.25
    for _ in range(4):
        accel_l = ext_force_l - disp_l * stiff_l - vel_l * damp_l
        accel_r = ext_force_r - disp_r * stiff_r - vel_r * damp_r

        vel_l.x = max(-1.5, min(1.5, vel_l.x + accel_l.x * sub_dt))
        vel_l.y = max(-1.5, min(1.5, vel_l.y + accel_l.y * sub_dt))
        vel_l.z = max(-1.5, min(1.5, vel_l.z + accel_l.z * sub_dt))

        vel_r.x = max(-1.5, min(1.5, vel_r.x + accel_r.x * sub_dt))
        vel_r.y = max(-1.5, min(1.5, vel_r.y + accel_r.y * sub_dt))
        vel_r.z = max(-1.5, min(1.5, vel_r.z + accel_r.z * sub_dt))

        disp_l.x = max(-0.024, min(0.024, disp_l.x + vel_l.x * sub_dt))
        disp_l.y = max(-0.024, min(0.024, disp_l.y + vel_l.y * sub_dt))
        disp_l.z = max(-0.024, min(0.024, disp_l.z + vel_l.z * sub_dt))

        disp_r.x = max(-0.024, min(0.024, disp_r.x + vel_r.x * sub_dt))
        disp_r.y = max(-0.024, min(0.024, disp_r.y + vel_r.y * sub_dt))
        disp_r.z = max(-0.024, min(0.024, disp_r.z + vel_r.z * sub_dt))

    state["disp_l"] = disp_l
    state["disp_r"] = disp_r
    state["vel_l"] = vel_l
    state["vel_r"] = vel_r

    return disp_l, disp_r


def _preview_frame_handler(scene):
    props = getattr(scene, "xxmi_custom_physics_props", None)
    if not props or not props.is_preview_running:
        return

    objects_data = _PREVIEW_CACHE.get("objects_data", [])
    if not objects_data:
        return

    fps = scene.render.fps / max(scene.render.fps_base, 1e-4)
    t = scene.frame_current / max(fps, 24.0)

    disp_l_local, disp_r_local = _step_preview_ode(t, props)
    strength = max(0.0, float(props.strength))
    norm_factor = _PREVIEW_CACHE.get("norm_factor", 1.0)
    scale_factor = norm_factor * strength
    max_disp = 0.024 * max(0.1, strength)

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
        rigid_flags = odata["rigid_flags"]
        anchor_l = odata["anchor_l"]
        anchor_r = odata["anchor_r"]
        up_vec = odata["up_vec"]
        right_vec = odata["right_vec"]
        fwd_vec = odata["fwd_vec"]

        if len(sk.data) != len(base_coords):
            continue

        disp_l = right_vec * disp_l_local.x + fwd_vec * disp_l_local.y + up_vec * disp_l_local.z
        disp_r = right_vec * disp_r_local.x + fwd_vec * disp_r_local.y + up_vec * disp_r_local.z

        # 인게임 CustomPhysics_{idx}.hlsl과 100% 동일한 가중치 선형 변위 적용
        for i in active_indices:
            base_co = base_coords[i]
            if scale_factor <= 0.0001:
                sk.data[i].co = base_co
                continue
            wl = w_l[i]
            wr = w_r[i]

            total_disp = (disp_l * wl + disp_r * wr) * scale_factor
            d_len = total_disp.length
            if d_len > max_disp and d_len > 1e-5:
                total_disp = (total_disp / d_len) * max_disp
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

        # 인게임 NORM_FACTOR와 100% 동일하게 전체 대상 메쉬의 최대 가중치로 정규화 배율 산출
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
    bl_label = "기본값으로 초기화"
    bl_description = "가슴 물리 및 관성 파라미터(강도, 속도, 감쇠, 관성, 호흡 모션)를 권장 기본값으로 되돌립니다"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.xxmi_custom_physics_props
        props.strength = 1.0
        props.speed = 1.1
        props.damping = 0.85
        props.inertia = 1.2
        props.idle_motion = 0.4
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
