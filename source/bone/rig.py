import bpy
from mathutils import Vector
from ..utils import ensure_object_visible

PHYSICS_RIG_NAME = "XXMI_Physics_Rig"
BREAST_RIG_NAME = PHYSICS_RIG_NAME
VG_BREAST_L = "XXMI_Breast_L"
VG_BREAST_R = "XXMI_Breast_R"


def get_or_create_physics_rig(context, props, ref_obj):
    """물리 앵커 역할을 하는 Armature(XXMI_Physics_Rig)를 가져오거나 새로 생성합니다."""
    rig = props.anchor_obj
    if rig and rig.name in bpy.data.objects and rig.type == 'ARMATURE':
        ensure_object_visible(context, rig)
        return rig

    for r_name in (PHYSICS_RIG_NAME, "XXMI_Breast_Rig"):
        existing = bpy.data.objects.get(r_name)
        if existing and existing.type == 'ARMATURE':
            props.anchor_obj = existing
            ensure_object_visible(context, existing)
            return existing

    arm_data = bpy.data.armatures.new(PHYSICS_RIG_NAME)
    arm_data.display_type = 'OCTAHEDRAL'

    rig = bpy.data.objects.new(PHYSICS_RIG_NAME, arm_data)
    rig.show_in_front = True
    col = ref_obj.users_collection[0] if (ref_obj and ref_obj.users_collection) else context.scene.collection
    col.objects.link(rig)
    props.anchor_obj = rig
    return rig


get_or_create_breast_rig = get_or_create_physics_rig


def get_rig_bone_info(rig):
    """
    XXMI_Breast_Rig 내의 본(Breast.L, Breast.R) 정보를 월드 좌표계로 추출합니다.
    """
    bone_info = {}
    if not rig or rig.type != 'ARMATURE':
        return bone_info

    if rig.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.view_layer.update()

    mw = rig.matrix_world
    pose_bones = rig.pose.bones if rig.pose else None
    data_bones = rig.data.bones

    for db in data_bones:
        bname = db.name
        pb = pose_bones.get(bname) if pose_bones else None
        if pb:
            head_w = mw @ pb.head
            tail_w = mw @ pb.tail
        else:
            head_w = mw @ db.head_local
            tail_w = mw @ db.tail_local

        fwd = tail_w - head_w
        fwd_len = fwd.length
        fwd_unit = (fwd / fwd_len) if fwd_len > 1e-4 else Vector((0.0, -1.0, 0.0))
        cent = head_w + fwd * 0.4
        tip = tail_w

        scale_factor = 1.0
        if pb:
            scale_factor = max(abs(pb.scale.x), abs(pb.scale.y), abs(pb.scale.z))

        r_cands = []
        if db.head_radius > 0.005:
            r_cands.append(db.head_radius * scale_factor)
        if db.envelope_distance > 0.005:
            r_cands.append(db.envelope_distance * scale_factor)
        if fwd_len > 0.01:
            r_cands.append(fwd_len * 0.85)

        radius = max(r_cands) if r_cands else 0.10
        radius = max(0.03, radius)

        entry = {
            "bname": bname,
            "centroid": cent,
            "tip": tip,
            "head": head_w,
            "tail": tail_w,
            "fwd_unit": fwd_unit,
            "radius": radius,
        }
        bone_info[bname] = entry

        b_lower = bname.lower()
        if b_lower in ("breast.l", "l"):
            bone_info["L"] = entry
        elif b_lower in ("breast.r", "r"):
            bone_info["R"] = entry

    # "L"/"R" 명칭이 아니더라도 X 좌표에 따라 1/2채널 자동 매핑
    if "L" not in bone_info and "R" not in bone_info:
        named_entries = [e for k, e in bone_info.items() if k not in ("L", "R")]
        if len(named_entries) >= 2:
            sorted_by_x = sorted(named_entries, key=lambda e: e["head"].x, reverse=True)
            bone_info["L"] = sorted_by_x[0]
            bone_info["R"] = sorted_by_x[1]
        elif len(named_entries) == 1:
            bone_info["L"] = named_entries[0]

    return bone_info

