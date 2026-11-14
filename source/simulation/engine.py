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
    "active_bones": [],
    "norm_factor": 1.0,
    "ode_state": {
        "bones": {},  # bname -> {"disp": Vector, "vel": Vector}
        "prev_t": None,
    },
}


def _simulate_bust_step(b_st, anim_vel, yaw_rate, roll_rate, side_sign, t, sub_dt, stiff, damp, inertia, idle, speed):
    """바스트 모핑 본(가슴) 전용 4-Substep ODE 물리 적분 (전후·상하 탄성 + 흉부 호흡)"""
    bone_vel = Vector((
        anim_vel.x,
        anim_vel.y - side_sign * yaw_rate * 0.45,
        anim_vel.z + side_sign * roll_rate * 0.35,
    ))

    ext_force = Vector((
        -bone_vel.x * 12.0 * inertia,
        -bone_vel.y * 12.0 * inertia,
        -bone_vel.z * 12.0 * inertia,
    ))

    if idle > 0.001:
        breath_phase = t * (2.8 * speed) + (0.0 if side_sign > 0 else 0.785398)
        ext_force.y += math.sin(breath_phase) * 1.5 * idle
        ext_force.z += math.cos(breath_phase) * 1.0 * idle

    disp = b_st["disp"]
    vel = b_st["vel"]

    for _ in range(4):
        accel = ext_force - disp * stiff - vel * damp

        vel.x = max(-2.5, min(2.5, vel.x + accel.x * sub_dt))
        vel.y = max(-2.5, min(2.5, vel.y + accel.y * sub_dt))
        vel.z = max(-2.5, min(2.5, vel.z + accel.z * sub_dt))

        disp.x = max(-0.035, min(0.035, disp.x + vel.x * sub_dt))
        disp.y = max(-0.035, min(0.035, disp.y + vel.y * sub_dt))
        disp.z = max(-0.035, min(0.035, disp.z + vel.z * sub_dt))

    b_st["disp"] = disp
    b_st["vel"] = vel
    return disp


def _simulate_hip_step(b_st, anim_vel, yaw_rate, roll_rate, side_sign, t, sub_dt, stiff, damp, inertia, idle, speed):
    """
    힙 모핑 본(엉덩이) 전용 4-Substep ODE 물리 적분:
    - 바스트보다 절제된 진폭과 비선형 조직 경화(Cubic Tissue Stiffening)로 과도한 출렁임 및 지방 압축/인장 왜곡 차단
    - 보행/스윙 속도를 절제된 상하·좌우 탄성으로 변환하고, 비압축 체적 보존 곡면 회전호(Incompressible Glute Arc) 적용
    """
    stride_y = anim_vel.y - side_sign * yaw_rate * 0.45
    vert_z = anim_vel.z + side_sign * roll_rate * 0.35

    ext_force = Vector((
        (-anim_vel.x * 0.75 + side_sign * stride_y * 0.20) * 8.5 * inertia,
        (-stride_y * 0.08) * 8.5 * inertia,
        (-vert_z * 0.85 + stride_y * 0.55) * 8.5 * inertia,
    ))

    if idle > 0.001:
        sway_phase = t * (2.0 * speed) + (0.0 if side_sign > 0 else 1.570796)
        ext_force.x += math.sin(sway_phase) * 0.35 * idle
        ext_force.z += math.cos(sway_phase) * 0.60 * idle

    disp = b_st["disp"]
    vel = b_st["vel"]

    stiff_base = stiff * 1.35
    damp_base = damp * 1.25
    stiff_y = stiff * 2.4
    damp_y = damp * 1.8

    for _ in range(4):
        # 인체 지방/피부 비선형 콜라겐 경화(Cubic Stiffening): 일정 변위 이상 늘어나거나 압축되지 않도록 복원력 급증
        nonlin = 1.0 + disp.length_squared * 2200.0
        accel_x = ext_force.x - disp.x * (stiff_base * nonlin) - vel.x * damp_base
        accel_y = ext_force.y - disp.y * (stiff_y * nonlin) - vel.y * damp_y
        accel_z = ext_force.z - disp.z * (stiff_base * nonlin) - vel.z * damp_base

        vel.x = max(-1.5, min(1.5, vel.x + accel_x * sub_dt))
        vel.y = max(-0.6, min(0.6, vel.y + accel_y * sub_dt))
        vel.z = max(-1.8, min(1.8, vel.z + accel_z * sub_dt))

        disp.x = max(-0.012, min(0.012, disp.x + vel.x * sub_dt))
        disp.y = max(-0.002, min(0.005, disp.y + vel.y * sub_dt))
        disp.z = max(-0.016, min(0.016, disp.z + vel.z * sub_dt))

    # 소프트 탄성 한계 포화(Soft Tissue Saturation) 및 비압축 체적 보존 곡면 호(Incompressible Glute Arc)
    raw_len = disp.length
    sat_disp = disp.copy()
    if raw_len > 1e-5:
        max_hip_disp = 0.015
        sat_len = max_hip_disp * (raw_len / (raw_len + max_hip_disp * 0.85))
        sat_disp = disp * (sat_len / raw_len)

    arc_disp = Vector((
        sat_disp.x,
        max(0.0, sat_disp.y * 0.25) + abs(sat_disp.z) * 0.16 - sat_disp.z * 0.08,
        sat_disp.z,
    ))
    b_st["disp"] = disp
    b_st["vel"] = vel
    return arc_disp


def _step_preview_ode(t, props, active_bones):
    """
    인게임 CustomPhysicsMotion.hlsl과 100% 동일한 4-Substep Symplectic Euler ODE 적분을 수행하여
    본 종류(바스트 모핑 vs 힙 모핑)에 따른 물리 변위 dict({bname: Vector})를 반환합니다.
    """
    state = _PREVIEW_CACHE["ode_state"]
    prev_t = state.get("prev_t")
    bones_state = state.setdefault("bones", {})

    if prev_t is None or t < prev_t or (t - prev_t) > 0.25:
        dt = 1.0 / 60.0
        if prev_t is None:
            for bn in active_bones:
                bones_state[bn] = {
                    "disp": Vector((0.0, 0.0, 0.0)),
                    "vel": Vector((0.0, 0.0, 0.0)),
                }
    else:
        raw_dt = t - prev_t
        if abs(raw_dt) <= 0.0005:
            return {bn: bones_state.get(bn, {}).get("disp", Vector((0.0, 0.0, 0.0))) for bn in active_bones}
        dt = max(0.002, min(0.04, raw_dt))

    state["prev_t"] = t

    # 3.0초 주기의 인게임 표준 몸통 모션 시뮬레이션 (선형 이동 + 토르소 회전 요/롤 복합)
    cycle = t % 3.0
    if cycle < 1.4:
        ph = (cycle / 1.4) * math.pi * 2.0
        raw_vel = Vector((
            math.cos(ph) * 0.14,
            math.sin(ph) * 0.15,
            math.sin(ph * 2.0) * 0.20,
        ))
        yaw_rate = math.sin(ph) * 0.12
        roll_rate = math.cos(ph) * 0.08
    else:
        raw_vel = Vector((0.0, 0.0, 0.0))
        yaw_rate = 0.0
        roll_rate = 0.0

    anim_vel = raw_vel / (1.0 + raw_vel.length * 0.5)

    disps_out = {}
    sub_dt = dt * 0.25

    for idx, bn in enumerate(active_bones):
        b_st = bones_state.setdefault(bn, {
            "disp": Vector((0.0, 0.0, 0.0)),
            "vel": Vector((0.0, 0.0, 0.0)),
        })

        params = get_bone_params(props, bn)
        btype = params.get("bone_type", "MORPH")
        speed = max(0.2, float(params["speed"]))
        damping = max(0.1, float(params["damping"]))
        inertia = max(0.0, float(params["inertia"]))
        idle = max(0.0, float(params["idle_motion"]))

        omega = 12.0 * speed
        stiff = omega * omega
        damp = 2.0 * omega * (damping * 0.55)

        b_low = bn.lower()
        is_right = ("r" in b_low or "오" in b_low or idx % 2 == 1)
        side_sign = -1.0 if is_right else 1.0

        if btype == 'HIP_MORPH':
            disps_out[bn] = _simulate_hip_step(
                b_st, anim_vel, yaw_rate, roll_rate, side_sign, t, sub_dt, stiff, damp, inertia, idle, speed
            )
        else:
            disps_out[bn] = _simulate_bust_step(
                b_st, anim_vel, yaw_rate, roll_rate, side_sign, t, sub_dt, stiff, damp, inertia, idle, speed
            )

    return disps_out


def preview_frame_handler(scene):
    props = getattr(scene, "xxmi_custom_physics_props", None)
    if not props or not props.is_preview_running:
        return

    objects_data = _PREVIEW_CACHE.get("objects_data", [])
    active_bones = _PREVIEW_CACHE.get("active_bones", [])
    if not objects_data or not active_bones:
        return

    fps = scene.render.fps / max(scene.render.fps_base, 1e-4)
    t = scene.frame_current / max(fps, 24.0)

    disps_by_bone = _step_preview_ode(t, props, active_bones)
    norm_factor = _PREVIEW_CACHE.get("norm_factor", 1.0)

    bone_scales = {}
    for bn in active_bones:
        p = get_bone_params(props, bn)
        st = max(0.0, float(p["strength"]))
        bone_scales[bn] = norm_factor * st

    max_strength = max([max(0.1, float(get_bone_params(props, bn)["strength"])) for bn in active_bones], default=1.0)
    max_disp = 0.035 * max_strength

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
        weights_by_bone = odata["weights_by_bone"]
        up_vec = odata["up_vec"]
        right_vec = odata["right_vec"]
        fwd_vec = odata["fwd_vec"]

        if len(sk.data) != len(base_coords):
            continue

        # 각 본별 3D 월드 변위 벡터 사전 계산
        disp_vectors = {}
        for bn in active_bones:
            disp_local = disps_by_bone.get(bn, Vector((0.0, 0.0, 0.0)))
            disp_vectors[bn] = right_vec * disp_local.x + fwd_vec * disp_local.y + up_vec * disp_local.z

        for i in active_indices:
            base_co = base_coords[i]
            total_disp = Vector((0.0, 0.0, 0.0))
            sum_eff = 0.0
            max_eff = 0.0
            for bn in active_bones:
                w_arr = weights_by_bone.get(bn)
                if w_arr:
                    eff_w = w_arr[i] * bone_scales[bn]
                    if eff_w > 1e-4:
                        total_disp += disp_vectors[bn] * eff_w
                        sum_eff += eff_w
                        if eff_w > max_eff:
                            max_eff = eff_w

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
    _PREVIEW_CACHE["active_bones"] = []

    if hasattr(context, "scene") and hasattr(context.scene, "xxmi_custom_physics_props"):
        context.scene.xxmi_custom_physics_props.is_preview_running = False
