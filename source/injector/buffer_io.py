import os
import json
import math
import numpy as np
import bpy
from mathutils import Vector, Matrix


def _read_buf_positions_np(buf_path):
    """Position.buf (stride 40)의 위치(xyz)만 (n, 3) float64 numpy 배열로 빠르게 읽습니다."""
    data = np.fromfile(buf_path, dtype='<f4')
    n = len(data) // 10
    return data[:n * 10].reshape(n, 10)[:, :3].astype(np.float64)


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


def _export_physics_state_buf(export_dir, idx):
    out_path = os.path.join(export_dir, f"CustomPhysicsState_{idx}.buf")
    with open(out_path, 'wb') as f:
        f.write(bytearray(5 * 16))


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
        for it in props.bone_items:
            bone_items_list.append({
                "name": it.name,
                "bone_type": it.bone_type,
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
            })

    cfg = {
        "version": "1.0",
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
    }

    out_path = os.path.join(export_dir, "XXMI_Physics_Config.json")
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"XXMI_Physics_Config.json 저장 실패: {e}")

