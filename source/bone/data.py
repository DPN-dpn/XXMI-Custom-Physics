import json
import bpy


def get_bone_vg_name(bname):
    """모핑 본의 버텍스 그룹 이름을 반환합니다. (규칙: XXMI_<본이름>)"""
    return f"XXMI_{bname}"


def handle_bone_item_rename(item, context=None):
    """
    모핑 본 슬롯의 이름(item.name)이 변경되었을 때:
    1. 관련 메쉬들의 버텍스 그룹 이름을 XXMI_<이전이름>에서 XXMI_<새이름>으로 일괄 변경
    2. 물리 리그(XXMI_Physics_Rig / XXMI_Breast_Rig) 내의 본 이름 변경
    3. 강체 슬롯(props.rigid_items)의 target_bone 참조 변경
    4. item.prev_name 갱신
    """
    if not item:
        return

    new_name = item.name.strip()
    if not new_name:
        return

    prev_name = getattr(item, "prev_name", "")
    if not prev_name:
        item.prev_name = new_name
        return

    if prev_name == new_name:
        return

    old_vg_name = get_bone_vg_name(prev_name)
    new_vg_name = get_bone_vg_name(new_name)

    # 1. 씬 내 모든 메쉬 오브젝트의 버텍스 그룹 이름 일괄 변경
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            vg = obj.vertex_groups.get(old_vg_name)
            if vg:
                vg.name = new_vg_name

    # 2. 물리 리그 내의 본 이름 변경
    props = getattr(bpy.context.scene, "xxmi_custom_physics_props", None) if bpy.context else None
    rig = None
    if props and props.anchor_obj:
        rig = props.anchor_obj
    else:
        rig = bpy.data.objects.get("XXMI_Physics_Rig") or bpy.data.objects.get("XXMI_Breast_Rig")

    if rig and rig.type == 'ARMATURE':
        was_mode = bpy.context.mode
        orig_active = bpy.context.view_layer.objects.active
        if was_mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

        try:
            bpy.context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode='EDIT')
            if prev_name in rig.data.edit_bones:
                rig.data.edit_bones[prev_name].name = new_name
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        if orig_active and orig_active.name in bpy.data.objects:
            try:
                bpy.context.view_layer.objects.active = orig_active
                if was_mode == 'EDIT_MESH':
                    bpy.ops.object.mode_set(mode='EDIT')
            except Exception:
                pass

    # 3. 강체 슬롯의 target_bone 참조 갱신
    if props:
        for r_item in props.rigid_items:
            if r_item.target_bone == prev_name:
                r_item.target_bone = new_name

    item.prev_name = new_name


def get_bone_data(item):
    """BoneItem의 JSON 문자열 데이터를 딕셔너리로 파싱합니다."""
    if not item or not item.data_json:
        return {}
    try:
        return json.loads(item.data_json)
    except Exception:
        return {}


def set_bone_data(item, data):
    """BoneItem의 JSON 문자열 데이터를 갱신하고 정점 수를 동기화합니다."""
    if not item:
        return
    item.data_json = json.dumps(data)
    item.vert_count = sum(len(v) for v in data.values())


def get_bone_item(props, bname):
    """본 이름 또는 버텍스 그룹 이름에 해당하는 슬롯을 반환합니다."""
    if not props or not hasattr(props, "bone_items"):
        return None

    b_lower = bname.lower()
    for item in props.bone_items:
        if item.name.lower() == b_lower:
            return item

    # XXMI_ 접두사가 붙어있는 경우 제거 후 검색
    if b_lower.startswith("xxmi_"):
        stripped = b_lower[5:]
        for item in props.bone_items:
            if item.name.lower() == stripped:
                return item

    return None


def get_bone_params(props, bname):
    """
    지정된 본의 개별 가중치 및 물리 파라미터를 dict로 반환합니다.
    해당 슬롯이 없으면 props의 전역 기본값으로 fallback합니다.
    """
    item = get_bone_item(props, bname)
    if item:
        return {
            "name": item.name,
            "peak_weight": float(item.peak_weight),
            "falloff_power": float(item.falloff_power),
            "smooth_iterations": int(item.smooth_iterations),
            "lock_registered_verts": bool(item.lock_registered_verts),
            "strength": float(item.strength),
            "speed": float(item.speed),
            "damping": float(item.damping),
            "inertia": float(item.inertia),
            "idle_motion": float(item.idle_motion),
        }

    return {
        "name": bname,
        "peak_weight": float(getattr(props, "peak_weight", 0.40)),
        "falloff_power": float(getattr(props, "falloff_power", 1.0)),
        "smooth_iterations": int(getattr(props, "smooth_iterations", 2)),
        "lock_registered_verts": bool(getattr(props, "lock_registered_verts", True)),
        "strength": float(getattr(props, "strength", 1.0)),
        "speed": float(getattr(props, "speed", 1.1)),
        "damping": float(getattr(props, "damping", 0.85)),
        "inertia": float(getattr(props, "inertia", 1.2)),
        "idle_motion": float(getattr(props, "idle_motion", 0.4)),
    }


def ensure_default_breast_bone_items(props):
    """
    props.bone_items가 비어있으면 기본 모핑 본 슬롯 쌍을 생성합니다.
    """
    if not props or not hasattr(props, "bone_items"):
        return

    if len(props.bone_items) == 0:
        item_1 = props.bone_items.add()
        item_1.name = "모핑 본 1"
        item_1.bone_type = 'MORPH'
        item_1.peak_weight = getattr(props, "peak_weight", 0.40)
        item_1.falloff_power = getattr(props, "falloff_power", 1.0)
        item_1.smooth_iterations = getattr(props, "smooth_iterations", 2)
        item_1.lock_registered_verts = getattr(props, "lock_registered_verts", True)
        item_1.strength = getattr(props, "strength", 1.0)
        item_1.speed = getattr(props, "speed", 1.1)
        item_1.damping = getattr(props, "damping", 0.85)
        item_1.inertia = getattr(props, "inertia", 1.2)
        item_1.idle_motion = getattr(props, "idle_motion", 0.4)

        item_2 = props.bone_items.add()
        item_2.name = "모핑 본 2"
        item_2.bone_type = 'MORPH'
        item_2.peak_weight = getattr(props, "peak_weight", 0.40)
        item_2.falloff_power = getattr(props, "falloff_power", 1.0)
        item_2.smooth_iterations = getattr(props, "smooth_iterations", 2)
        item_2.lock_registered_verts = getattr(props, "lock_registered_verts", True)
        item_2.strength = getattr(props, "strength", 1.0)
        item_2.speed = getattr(props, "speed", 1.1)
        item_2.damping = getattr(props, "damping", 0.85)
        item_2.inertia = getattr(props, "inertia", 1.2)
        item_2.idle_motion = getattr(props, "idle_motion", 0.4)
