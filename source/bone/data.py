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
    """본 이름 또는 인덱스에 해당하는 슬롯을 반환합니다."""
    if not props or not hasattr(props, "bone_items"):
        return None

    b_lower = bname.lower()
    for item in props.bone_items:
        if item.name.lower() == b_lower:
            return item

    # 하위 호환용 (L/Breast.L -> 0번째 본, R/Breast.R -> 1번째 본)
    if b_lower in ("breast.l", "l") and len(props.bone_items) > 0:
        return props.bone_items[0]
    if b_lower in ("breast.r", "r") and len(props.bone_items) > 1:
        return props.bone_items[1]

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
