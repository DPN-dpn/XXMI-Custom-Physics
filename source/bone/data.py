import json
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import world_vertex_positions


def get_bone_vg_name(bname):
    """모핑 본의 버텍스 그룹 이름을 반환합니다. (규칙: XXMI_<본이름>)"""
    return f"XXMI_{bname}"


def handle_bone_item_rename(item, context=None):
    """
    모핑 본 슬롯의 이름(item.name)이 변경되었을 때:
    1. 관련 메쉬들의 버텍스 그룹 이름을 XXMI_<이전이름>에서 XXMI_<새이름>으로 일괄 변경
    2. 물리 리그(XXMI_Physics_Rig / XXMI_Breast_Rig) 내의 본 이름 변경
    3. item.prev_name 갱신
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
        from ..utils import ensure_object_visible
        was_mode = bpy.context.mode
        orig_active = bpy.context.view_layer.objects.active
        ensure_object_visible(bpy.context, rig)
        if was_mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass

        try:
            bpy.context.view_layer.objects.active = rig
            bpy.ops.object.mode_set(mode='EDIT')
            target_eb = rig.data.edit_bones.get(prev_name)
            if not target_eb:
                p_low = prev_name.lower()
                for eb in rig.data.edit_bones:
                    if eb.name.lower() == p_low:
                        target_eb = eb
                        break
            if target_eb:
                target_eb.name = new_name
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        if orig_active and orig_active.name in bpy.data.objects:
            try:
                ensure_object_visible(bpy.context, orig_active)
                bpy.context.view_layer.objects.active = orig_active
                if was_mode == 'EDIT_MESH':
                    bpy.ops.object.mode_set(mode='EDIT')
            except Exception:
                pass

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


def apply_bone_type_defaults(target, bone_type='MORPH'):
    """지정된 본 종류('MORPH' 바스트 모핑 vs 'HIP_MORPH' 힙 모핑)에 최적화된 가중치 및 물리 초기값을 적용합니다."""
    if bone_type == 'HIP_MORPH':
        # 힙 모핑: 바스트보다 절제된 진폭과 높은 감쇠, 넓은 일체형 볼륨 분포로 과도한 출렁임 및 인장/압축 왜곡 방지
        target.peak_weight = 0.32
        target.falloff_power = 0.70
        target.smooth_iterations = 4
        target.lock_registered_verts = True
        target.strength = 0.65
        target.speed = 1.35
        target.damping = 1.05
        target.inertia = 0.90
        target.idle_motion = 0.05
    else:
        target.peak_weight = 0.40
        target.falloff_power = 1.00
        target.smooth_iterations = 2
        target.lock_registered_verts = True
        target.strength = 1.00
        target.speed = 1.10
        target.damping = 0.85
        target.inertia = 1.20
        target.idle_motion = 0.40


def get_bone_params(props, bname):
    """
    지정된 본의 개별 가중치 및 물리 파라미터를 dict로 반환합니다.
    해당 슬롯이 없으면 props의 전역 기본값으로 fallback합니다.
    """
    item = get_bone_item(props, bname)
    if item:
        return {
            "name": getattr(item, "name", bname),
            "bone_type": str(getattr(item, "bone_type", "MORPH")),
            "radius_scale": float(getattr(item, "radius_scale", getattr(props, "radius_scale", 1.15))),
            "peak_weight": float(getattr(item, "peak_weight", getattr(props, "peak_weight", 0.40))),
            "falloff_power": float(getattr(item, "falloff_power", getattr(props, "falloff_power", 1.0))),
            "smooth_iterations": int(getattr(item, "smooth_iterations", getattr(props, "smooth_iterations", 2))),
            "lock_registered_verts": bool(getattr(item, "lock_registered_verts", getattr(props, "lock_registered_verts", True))),
            "strength": float(getattr(item, "strength", getattr(props, "strength", 0.35))),
            "speed": float(getattr(item, "speed", getattr(props, "speed", 2.5))),
            "damping": float(getattr(item, "damping", getattr(props, "damping", 0.05))),
            "inertia": float(getattr(item, "inertia", getattr(props, "inertia", 0.40))),
            "idle_motion": float(getattr(item, "idle_motion", getattr(props, "idle_motion", 0.02))),
        }

    return {
        "name": bname,
        "bone_type": str(getattr(props, "bone_type", "MORPH")),
        "radius_scale": float(getattr(props, "radius_scale", 1.15)),
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


def get_all_rigid_verts_by_obj(props):
    """등록된 모든 강체 슬롯에서 강체 정점 목록을 오브젝트별 set으로 수집합니다."""
    rigid_verts_by_obj = {}
    from ..rigid.data import get_rigid_data
    for r_item in props.rigid_items:
        r_dict = get_rigid_data(r_item)
        if not r_dict:
            continue
        for oname, r_vis in r_dict.items():
            rigid_verts_by_obj.setdefault(oname, set()).update(r_vis)
    return rigid_verts_by_obj


def subtract_rigid_from_bone_slot_data(item, rigid_verts_by_obj, keep_boundary=True):
    """
    물리 본 슬롯 item의 정점 목록에서 등록된 강체 정점들을 제외합니다.
    keep_boundary=True일 경우, 비-강체(물리 메인 표면) 정점과 엣지로 직접 연결되어 있거나
    공간상 맞닿아 있는 경계 접합 정점은 제외하지 않고 유지합니다.

    반환값: (subtracted_count, kept_boundary_count)
    """
    data = get_bone_data(item)
    if not data:
        return 0, 0

    new_data = {}
    subtracted_count = 0
    kept_boundary_count = 0

    for obj_name, v_indices in data.items():
        obj = bpy.data.objects.get(obj_name)
        if not obj or obj.type != 'MESH':
            new_data[obj_name] = v_indices
            continue

        rigid_set = rigid_verts_by_obj.get(obj_name, set())
        slot_set = set(v_indices)

        overlap = slot_set & rigid_set
        if not overlap:
            new_data[obj_name] = v_indices
            continue

        mesh = obj.data

        boundary_verts = set()
        if keep_boundary:
            # 1. 엣지 토폴로지 연결 검사: overlap 정점 중 비-강체 정점과 엣지로 연결된 정점 유지
            edge_neighbors = {v: [] for v in overlap}
            for e in mesh.edges:
                v1, v2 = e.vertices[0], e.vertices[1]
                if v1 in edge_neighbors:
                    edge_neighbors[v1].append(v2)
                if v2 in edge_neighbors:
                    edge_neighbors[v2].append(v1)

            for v, nbrs in edge_neighbors.items():
                if any(nbr not in rigid_set for nbr in nbrs):
                    boundary_verts.add(v)

            # 2. 메쉬가 분리되어 있어 엣지가 없는 경우를 위한 공간 근접성(KDTree) 보정
            non_rigid_slot = [v for v in slot_set if v not in rigid_set]
            if non_rigid_slot and (len(boundary_verts) < len(overlap)):
                wpos = world_vertex_positions(obj)
                kd = KDTree(len(non_rigid_slot))
                for idx, vi in enumerate(non_rigid_slot):
                    kd.insert(Vector(wpos[vi]), idx)
                kd.balance()

                for v in overlap:
                    if v not in boundary_verts:
                        pt = Vector(wpos[v])
                        hits = kd.find_range(pt, 0.0015)
                        if hits:
                            boundary_verts.add(v)

        # 최종 남길 정점: (slot_set - overlap) + boundary_verts
        to_remove = overlap - boundary_verts
        remaining = [v for v in v_indices if v not in to_remove]

        new_data[obj_name] = remaining
        subtracted_count += len(to_remove)
        kept_boundary_count += len(boundary_verts)

        # 오브젝트의 버텍스 그룹에서도 제거된 정점 제외
        vg_name = get_bone_vg_name(item.name)
        vg = obj.vertex_groups.get(vg_name)
        if vg and to_remove:
            vg.remove(list(to_remove))

    set_bone_data(item, new_data)
    return subtracted_count, kept_boundary_count


def ensure_default_bone_items(props):
    """
    props.bone_items가 비어있으면 기본 물리 본 슬롯 쌍을 생성합니다.
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


ensure_default_breast_bone_items = ensure_default_bone_items

