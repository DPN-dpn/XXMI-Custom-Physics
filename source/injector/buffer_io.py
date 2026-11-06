import os
import json
import math
import numpy as np
import bpy
from mathutils import Vector, Matrix


def _get_addon_version():
    try:
        import sys
        # 1. XXMI-Custom-Physics 네임스페이스 모듈에서 bl_info 확인
        for name, m in sys.modules.items():
            if "XXMI-Custom-Physics" in name or "xxmi_custom_physics" in name.lower():
                if hasattr(m, "bl_info") and isinstance(m.bl_info, dict) and "version" in m.bl_info:
                    return ".".join(map(str, m.bl_info["version"]))

        # 2. 루트 __init__.py 파일 직접 파싱
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        init_file = os.path.join(root_dir, "__init__.py")
        if os.path.exists(init_file):
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            import re
            m = re.search(r'"version"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', content)
            if m:
                return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"

        # 3. 상대 import 시도
        from ... import bl_info
        if "version" in bl_info:
            return ".".join(map(str, bl_info["version"]))
    except Exception:
        pass
    return "1.0.1"


def _read_buf_positions_np(buf_path):
    """Position.buf (stride 40)의 위치(xyz)만 (n, 3) float64 numpy 배열로 빠르게 읽습니다."""
    data = np.fromfile(buf_path, dtype='<f4')
    n = len(data) // 10
    return data[:n * 10].reshape(n, 10)[:, :3].astype(np.float64)


def _read_buf_positions_and_normals_np(buf_path):
    """Position.buf (stride 40)의 위치(xyz)와 법선(normal)을 각각 (n, 3) float64 numpy 배열로 읽습니다."""
    data = np.fromfile(buf_path, dtype='<f4')
    n = len(data) // 10
    reshaped = data[:n * 10].reshape(n, 10)
    pos = reshaped[:, :3].astype(np.float64)
    norm = reshaped[:, 3:6].astype(np.float64)
    return pos, norm


def _read_buf_positions(buf_path):
    return [Vector(p) for p in _read_buf_positions_np(buf_path).tolist()]


def _find_best_transform_for_object(kd_buf, sample_obj, obj):
    """
    블렌더 오브젝트(로컬 좌표)의 가중치 정점 샘플이 Position.buf 정점들과 가장 잘 겹치는 변환 행렬을 추정합니다.
    """
    mw = obj.matrix_world
    rot_x_neg90 = Matrix.Rotation(math.radians(-90), 4, 'X')
    rot_x_pos90 = Matrix.Rotation(math.radians(90), 4, 'X')
    scale_100 = Matrix.Scale(100.0, 4)
    scale_001 = Matrix.Scale(0.01, 4)

    candidate_mats = [
        Matrix.Identity(4),
        mw,
        rot_x_neg90,
        rot_x_pos90,
        rot_x_neg90 @ mw,
        rot_x_pos90 @ mw,
    ]

    for s_mat in [scale_100, scale_001]:
        candidate_mats.append(s_mat)
        candidate_mats.append(s_mat @ mw)
        candidate_mats.append(rot_x_neg90 @ s_mat)
        candidate_mats.append(rot_x_pos90 @ s_mat)

    best_mat = None
    best_score = 999.0

    for mat in candidate_mats:
        dists = []
        for op in sample_obj:
            _, _, dist = kd_buf.find(mat @ op)
            dists.append(dist)

        dists.sort()
        p80_idx = max(0, int(len(dists) * 0.8) - 1)
        score = dists[p80_idx]
        if score < best_score:
            best_score = score
            best_mat = mat
            if score < 1e-4:
                break

    return best_mat, best_score


def _export_physics_state_buf(export_dir, idx, num_bones=1):
    out_path = os.path.join(export_dir, f"CustomPhysicsState_{idx}.buf")
    total_slots = max(1, num_bones) * 3
    with open(out_path, 'wb') as f:
        f.write(bytearray(total_slots * 16))


def _export_physics_config_json(export_dir, props, context):
    """모드 폴더에 현재 물리 파라미터 및 본 앵커 좌표를 담은 XXMI_Physics_Config.json을 저장합니다."""
    rig = props.anchor_obj or bpy.data.objects.get("XXMI_Physics_Rig") or bpy.data.objects.get("XXMI_Breast_Rig")
    bones_dict = {}

    if rig and rig.type == 'ARMATURE':
        mw = rig.matrix_world
        data_bones = rig.data.bones
        for db in data_bones:
            bname = db.name
            head_w = mw @ db.head_local
            tail_w = mw @ db.tail_local
            radius = max(0.03, db.head_radius, db.envelope_distance / 1.4)
            bones_dict[bname] = {
                "head": [round(head_w.x, 6), round(head_w.y, 6), round(head_w.z, 6)],
                "tail": [round(tail_w.x, 6), round(tail_w.y, 6), round(tail_w.z, 6)],
                "radius": round(radius, 6),
            }

    bone_items_list = []
    if hasattr(props, "bone_items"):
        from ..bone.data import get_bone_data
        for it in props.bone_items:
            bone_items_list.append({
                "name": it.name,
                "bone_type": it.bone_type,
                "chain_segments": int(getattr(it, "chain_segments", 4)),
                "gravity_factor": round(float(getattr(it, "gravity_factor", 1.0)), 4),
                "stiffness": round(float(getattr(it, "stiffness", 0.5)), 4),
                "peak_weight": round(float(it.peak_weight), 4),
                "falloff_power": round(float(it.falloff_power), 4),
                "smooth_iterations": int(it.smooth_iterations),
                "lock_registered_verts": bool(it.lock_registered_verts),
                "strength": round(float(it.strength), 4),
                "speed": round(float(it.speed), 4),
                "damping": round(float(it.damping), 4),
                "inertia": round(float(it.inertia), 4),
                "idle_motion": round(float(it.idle_motion), 4),
                "vert_count": int(it.vert_count),
                "data": get_bone_data(it),
            })

    rigid_items_list = []
    if hasattr(props, "rigid_items"):
        from ..rigid.data import get_rigid_data
        for rit in props.rigid_items:
            rigid_items_list.append({
                "name": rit.name,
                "weight_scale": round(float(rit.weight_scale), 4),
                "vert_count": int(rit.vert_count),
                "data": get_rigid_data(rit),
            })

    cfg = {
        "version": _get_addon_version(),
        "target_game": getattr(props, "target_game", "ZZZ"),
        "strength": round(float(props.strength), 4),
        "speed": round(float(props.speed), 4),
        "damping": round(float(props.damping), 4),
        "inertia": round(float(props.inertia), 4),
        "idle_motion": round(float(props.idle_motion), 4),
        "falloff_power": round(float(props.falloff_power), 4),
        "debug_mode": bool(props.debug_mode),
        "debug_toggle_key": str(props.debug_toggle_key),
        "bones": bones_dict,
        "bone_items": bone_items_list,
        "rigid_items": rigid_items_list,
    }

    out_path = os.path.join(export_dir, "XXMI_Physics_Config.json")
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"XXMI_Physics_Config.json 저장 실패: {e}")

