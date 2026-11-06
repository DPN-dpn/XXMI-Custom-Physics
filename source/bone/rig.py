import bpy
from mathutils import Vector
from ..utils import (
    ensure_object_visible,
    ensure_all_objects_in_object_mode,
    world_vertex_positions,
    compute_density_invariant_geometric_center,
)
from .data import get_bone_data

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
    arm_data.display_type = 'ENVELOPE' if getattr(props, "show_envelope_gizmo", False) else 'OCTAHEDRAL'

    rig = bpy.data.objects.new(PHYSICS_RIG_NAME, arm_data)
    rig.show_in_front = True
    col = ref_obj.users_collection[0] if (ref_obj and ref_obj.users_collection) else context.scene.collection
    col.objects.link(rig)
    props.anchor_obj = rig
    return rig


get_or_create_breast_rig = get_or_create_physics_rig


def get_rig_bone_info(rig):
    """
    XXMI_Physics_Rig 내의 모든 물리 본 정보를 월드 좌표계로 추출합니다.
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

        base_r = db.get("xxmi_base_radius", None)
        if base_r is not None:
            r_scale = 1.15
            props = getattr(bpy.context.scene, "xxmi_custom_physics_props", None) if bpy.context else None
            if props and hasattr(props, "bone_items"):
                for it in props.bone_items:
                    if it.name.lower() == bname.lower():
                        r_scale = max(0.2, float(getattr(it, "radius_scale", 1.15)))
                        break
            radius = float(base_r) * r_scale * scale_factor
        elif db.envelope_distance > 0.005:
            radius = db.envelope_distance * scale_factor
        elif db.head_radius > 0.005:
            radius = db.head_radius * scale_factor
        elif fwd_len > 0.01:
            radius = fwd_len * 0.70
        else:
            radius = 0.08
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


def sync_slot_bone_to_rig(context, item):
    """
    지정된 물리 본 슬롯 item의 정점 데이터를 기반으로
    XXMI_Physics_Rig 내의 해당 본을 생성하거나 위치/크기를 자동 갱신합니다.
    본 종류(MORPH, TAIL, HAIR, SKIRT, WING)에 따라 적합한 본 구조를 배치합니다.
    """
    data = get_bone_data(item)
    if not data:
        return None

    # 해당 슬롯의 모든 정점 월드 좌표 수집
    pts = []
    target_objs = []
    for oname, vis in data.items():
        obj = bpy.data.objects.get(oname)
        if obj and obj.type == 'MESH' and vis:
            target_objs.append(obj)
            w_arr = world_vertex_positions(obj)
            for vi in vis:
                if vi < len(w_arr):
                    pts.append(w_arr[vi])

    if len(pts) < 4:
        return None

    props = context.scene.xxmi_custom_physics_props
    ref_obj = target_objs[0]
    rig = get_or_create_physics_rig(context, props, ref_obj)
    ensure_object_visible(context, rig)

    # 정점들의 형태 중심 및 바운딩 박스 계산
    center, bmin, bmax = compute_density_invariant_geometric_center(pts)
    depth = max(0.04, bmax.y - bmin.y)
    span_x = max(0.04, bmax.x - bmin.x)
    span_z = max(0.04, bmax.z - bmin.z)
    base_radius = max(0.045, span_x * 0.50, span_z * 0.50)
    r_scale = max(0.2, float(getattr(item, "radius_scale", 1.15)))
    bone_radius = base_radius * r_scale

    dx_out = 0.05 * center.x if abs(center.x) > 0.02 else 0.0
    head = Vector((center.x, bmax.y, center.z))
    tail = Vector((center.x + dx_out, bmin.y, center.z))
    if (tail - head).length < 0.04:
        tail = Vector((center.x + dx_out, head.y - 0.04, center.z))

    was_mode = context.mode
    active_obj = context.view_layer.objects.active

    if was_mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')

    ebones = rig.data.edit_bones
    btype = getattr(item, "bone_type", 'MORPH')

    if btype in {'TAIL', 'HAIR'}:
        # 다관절 체인 (순차 연결 FK 본) 생성
        n_segs = max(1, getattr(item, "chain_segments", 4))
        vec_delta = (tail - head) / n_segs
        prev_bone = None
        for si in range(n_segs):
            s_name = item.name if si == 0 else f"{item.name}.{si:03d}"
            eb = ebones.get(s_name) or ebones.new(s_name)
            s_head = head + vec_delta * si
            s_tail = head + vec_delta * (si + 1)
            eb.head = s_head
            eb.tail = s_tail
            seg_base = base_radius * (1.0 - 0.4 * (si / max(1, n_segs)))
            seg_radius = seg_base * r_scale
            eb["xxmi_base_radius"] = float(seg_base)
            eb.head_radius = seg_radius
            eb.tail_radius = seg_radius * 0.8
            eb.envelope_distance = seg_radius * 1.05
            eb["xxmi_bone_type"] = btype
            if prev_bone:
                eb.parent = prev_bone
                eb.use_connect = True
            prev_bone = eb
        bone_out_name = item.name
    else:
        eb = ebones.get(item.name)
        if not eb:
            eb = ebones.new(item.name)

        eb.head = head
        eb.tail = tail
        eb["xxmi_base_radius"] = float(base_radius)
        eb.head_radius = bone_radius
        eb.tail_radius = bone_radius * 0.75
        eb.envelope_distance = bone_radius * 1.05
        eb["xxmi_bone_type"] = btype

        bone_out_name = eb.name

    bpy.ops.object.mode_set(mode='OBJECT')
    rig.show_in_front = True

    # 원래 모드로 복귀 (만약 에디트 모드였다면 원래 오브젝트 선택)
    if was_mode == 'EDIT_MESH' and active_obj:
        context.view_layer.objects.active = active_obj
        bpy.ops.object.mode_set(mode='EDIT')

    return bone_out_name


def update_rig_bone_radius_from_slot(context, item):
    """
    1번 패널에서 본 반경 배율(radius_scale) 슬라이더를 조절할 때
    실시간으로 XXMI_Physics_Rig 내 해당 본의 head_radius, tail_radius, envelope_distance를 갱신합니다.
    """
    if not item:
        return
    props = getattr(context.scene, "xxmi_custom_physics_props", None) if context else None
    rig = (getattr(props, "anchor_obj", None) if props else None) or bpy.data.objects.get(PHYSICS_RIG_NAME)
    if not rig or rig.type != 'ARMATURE':
        return

    r_scale = max(0.2, float(getattr(item, "radius_scale", 1.15)))
    bname_low = item.name.strip().lower()

    if rig.mode == 'EDIT':
        for eb in rig.data.edit_bones:
            eb_low = eb.name.lower()
            if eb_low == bname_low or eb_low.startswith(f"{bname_low}."):
                base_r = eb.get("xxmi_base_radius", None)
                if base_r is None:
                    base_r = max(0.045, eb.head_radius / 1.15)
                    eb["xxmi_base_radius"] = float(base_r)
                seg_r = float(base_r) * r_scale
                eb.head_radius = seg_r
                eb.tail_radius = seg_r * 0.75
                eb.envelope_distance = seg_r * 1.05
    else:
        for db in rig.data.bones:
            db_low = db.name.lower()
            if db_low == bname_low or db_low.startswith(f"{bname_low}."):
                base_r = db.get("xxmi_base_radius", None)
                if base_r is None:
                    base_r = max(0.045, db.head_radius / 1.15)
                    db["xxmi_base_radius"] = float(base_r)
                seg_r = float(base_r) * r_scale
                db.head_radius = seg_r
                db.tail_radius = seg_r * 0.75
                db.envelope_distance = seg_r * 1.05


def reposition_all_slot_bones(context, props):
    """
    모든 물리 본 슬롯의 정점 형태 중심을 분석하여 XXMI_Physics_Rig 내에 일괄 배치합니다.
    """
    if not props or not hasattr(props, "bone_items") or len(props.bone_items) == 0:
        return 0

    synced_count = 0
    for item in props.bone_items:
        res = sync_slot_bone_to_rig(context, item)
        if res:
            synced_count += 1

    rig = getattr(props, "anchor_obj", None) or bpy.data.objects.get(PHYSICS_RIG_NAME)
    if rig:
        rig.show_in_front = True
        context.view_layer.objects.active = rig
        rig.select_set(True)

    return synced_count
