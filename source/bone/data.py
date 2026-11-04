import json
import bpy


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
    """본 이름 또는 접미사(Breast.L, L, Breast.R, R 등)에 해당하는 슬롯을 반환합니다."""
    if not props or not hasattr(props, "bone_items"):
        return None

    b_lower = bname.lower()
    for item in props.bone_items:
        if item.name.lower() == b_lower:
            return item
        if b_lower in ("breast.l", "l") and item.bone_type == 'BREAST_L':
            return item
        if b_lower in ("breast.r", "r") and item.bone_type == 'BREAST_R':
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
    props.bone_items에 가슴 본 쌍(Breast.L, Breast.R)이 없으면 기본 슬롯을 생성합니다.
    """
    if not props or not hasattr(props, "bone_items"):
        return

    has_l = any(it.bone_type == 'BREAST_L' or it.name.lower() == 'breast.l' for it in props.bone_items)
    has_r = any(it.bone_type == 'BREAST_R' or it.name.lower() == 'breast.r' for it in props.bone_items)

    if not has_l:
        item_l = props.bone_items.add()
        item_l.name = "Breast.L"
        item_l.bone_type = 'BREAST_L'
        item_l.peak_weight = getattr(props, "peak_weight", 0.40)
        item_l.falloff_power = getattr(props, "falloff_power", 1.0)
        item_l.smooth_iterations = getattr(props, "smooth_iterations", 2)
        item_l.lock_registered_verts = getattr(props, "lock_registered_verts", True)
        item_l.strength = getattr(props, "strength", 1.0)
        item_l.speed = getattr(props, "speed", 1.1)
        item_l.damping = getattr(props, "damping", 0.85)
        item_l.inertia = getattr(props, "inertia", 1.2)
        item_l.idle_motion = getattr(props, "idle_motion", 0.4)

    if not has_r:
        item_r = props.bone_items.add()
        item_r.name = "Breast.R"
        item_r.bone_type = 'BREAST_R'
        item_r.peak_weight = getattr(props, "peak_weight", 0.40)
        item_r.falloff_power = getattr(props, "falloff_power", 1.0)
        item_r.smooth_iterations = getattr(props, "smooth_iterations", 2)
        item_r.lock_registered_verts = getattr(props, "lock_registered_verts", True)
        item_r.strength = getattr(props, "strength", 1.0)
        item_r.speed = getattr(props, "speed", 1.1)
        item_r.damping = getattr(props, "damping", 0.85)
        item_r.inertia = getattr(props, "inertia", 1.2)
        item_r.idle_motion = getattr(props, "idle_motion", 0.4)
