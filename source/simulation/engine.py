import bpy
import math
from mathutils import Vector
from ..utils import (
    PREVIEW_SK_NAME,
    PREVIEW_BASIS_TAG,
)
from ..bone.data import get_bone_params

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

    params_l = get_bone_params(props, "Breast.L")
    params_r = get_bone_params(props, "Breast.R")

    speed_l = max(0.2, float(params_l["speed"]))
    damping_l = max(0.1, float(params_l["damping"]))
    inertia_l = max(0.0, float(params_l["inertia"]))
    idle_l = max(0.0, float(params_l["idle_motion"]))

    speed_r = max(0.2, float(params_r["speed"]))
    damping_r = max(0.1, float(params_r["damping"]))
    inertia_r = max(0.0, float(params_r["inertia"]))
    idle_r = max(0.0, float(params_r["idle_motion"]))

    omega_l = 12.0 * speed_l
    stiff_l = omega_l * omega_l
    damp_l = 2.0 * omega_l * (damping_l * 0.50)

    omega_r = 12.0 * speed_r
    stiff_r = omega_r * omega_r
    damp_r = 2.0 * omega_r * (damping_r * 0.50)

    # 3.0초 주기의 인게임 표준 몸통 모션 시뮬레이션
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
    ext_force_l = -anim_vel * (12.0 * inertia_l)
    ext_force_r = -Vector((-anim_vel.x * 0.9, anim_vel.y, anim_vel.z)) * (12.0 * inertia_r)

    if idle_l > 0.001:
        breath_phase = t * 2.5
        ext_force_l.z += math.sin(breath_phase) * 1.5 * idle_l
        ext_force_l.y += math.cos(breath_phase) * 1.1 * idle_l
        ext_force_l.x += math.sin(breath_phase * 0.5) * 0.6 * idle_l

    if idle_r > 0.001:
        breath_phase = t * 2.5
        ext_force_r.z += math.sin(breath_phase + 0.35) * 1.5 * idle_r
        ext_force_r.y += math.cos(breath_phase + 0.35) * 1.1 * idle_r
        ext_force_r.x += -math.sin(breath_phase * 0.5 + 0.35) * 0.6 * idle_r

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


def preview_frame_handler(scene):
    props = getattr(scene, "xxmi_custom_physics_props", None)
    if not props or not props.is_preview_running:
        return

    objects_data = _PREVIEW_CACHE.get("objects_data", [])
    if not objects_data:
        return

    fps = scene.render.fps / max(scene.render.fps_base, 1e-4)
    t = scene.frame_current / max(fps, 24.0)

    disp_l_local, disp_r_local = _step_preview_ode(t, props)

    params_l = get_bone_params(props, "Breast.L")
    params_r = get_bone_params(props, "Breast.R")
    strength_l = max(0.0, float(params_l["strength"]))
    strength_r = max(0.0, float(params_r["strength"]))

    norm_factor = _PREVIEW_CACHE.get("norm_factor", 1.0)
    scale_l = norm_factor * strength_l
    scale_r = norm_factor * strength_r
    max_disp = max(0.024 * max(0.1, strength_l), 0.024 * max(0.1, strength_r))

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

        disp_l = right_vec * disp_l_local.x + fwd_vec * disp_l_local.y + up_vec * disp_l_local.z
        disp_r = right_vec * disp_r_local.x + fwd_vec * disp_r_local.y + up_vec * disp_r_local.z

        for i in active_indices:
            base_co = base_coords[i]
            wl = w_l[i]
            wr = w_r[i]
            if (wl * scale_l <= 0.0001) and (wr * scale_r <= 0.0001):
                sk.data[i].co = base_co
                continue

            total_disp = disp_l * (wl * scale_l) + disp_r * (wr * scale_r)
            d_len = total_disp.length
            if d_len > max_disp and d_len > 1e-5:
                total_disp = (total_disp / d_len) * max_disp
            sk.data[i].co = base_co + total_disp


def stop_physics_preview(context):
    """실행 중인 뷰포트 프리뷰를 중지하고 모든 대상 오브젝트의 쉐이프키를 완벽히 정리합니다."""
    if preview_frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(preview_frame_handler)

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

