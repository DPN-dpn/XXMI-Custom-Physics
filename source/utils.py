import bpy
from mathutils import Vector, Matrix

PHYSICS_VG_NAME = "XXMI_Physics"
ROLE_RIGID_VG = "XXMI_Role_Rigid"
ROLE_STATIC_VG = "XXMI_Role_Static"
BONE_VG_PREFIX = "XXMI_PB_"
RIG_DEFAULT_NAME = "XXMI_Physics_Rig"
PREVIEW_SK_NAME = "XXMI_Physics_Preview"
PREVIEW_BASIS_TAG = "_xxmi_temp_basis"
MERGE_MAT_PREFIX = "XXMI_Part_"
MERGE_META_KEY = "xxmi_merged_parts_meta"

BONE_TYPE_MAP = {
    'BODY': 0,    # 신체 지지 본 (팔, 다리, 몸통, 머리 등 비물리 고정 본)
    'BREAST': 1,  # 가슴 (흉곽 호흡 + 좌/우 관절 바운스)
    'HAIR': 2,    # 머리카락 / 포니테일 / 트윈테일 / 리본
    'SKIRT': 3,   # 스커트 / 코트 / 망토
    'TAIL': 4,    # 꼬리
    'WING': 5,    # 날개
}


def ensure_object_visible(context, obj):
    """AGENTS.md 컨벤션: 뷰포트 숨김/선택 불가 상태를 해제합니다."""
    if not obj:
        return
    if obj.hide_get():
        obj.hide_set(False)
    if obj.hide_select:
        obj.hide_select = False
    if hasattr(obj, "hide_viewport") and obj.hide_viewport:
        obj.hide_viewport = False


def ensure_all_objects_in_object_mode(context=None):
    """
    씬 내의 모든 오브젝트(메쉬, 아마추어 등)의 편집 모드를 안전하게 해제하여
    완벽히 OBJECT 모드로 전환합니다. (멀티 오브젝트 편집 모드로 인한 VertexGroup 런타임 오류 방지)
    """
    if context is None:
        context = bpy.context

    if context.mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

    for obj in bpy.data.objects:
        if obj.mode != 'OBJECT':
            try:
                ensure_object_visible(context, obj)
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass


def get_target_mesh_objects(context):
    """현재 선택되었거나 에디트 모드에 진입해 있는 모든 메쉬 오브젝트 목록을 반환합니다."""
    objs = []
    if context.mode == 'EDIT_MESH':
        objs = [o for o in context.objects_in_mode if o.type == 'MESH']
    if not objs:
        objs = [o for o in context.selected_objects if o.type == 'MESH']
    if not objs and context.active_object and context.active_object.type == 'MESH':
        objs = [context.active_object]
    return objs


def read_vertex_group_arrays(obj, group_map):
    """
    {그룹 인덱스: 키} 매핑에 해당하는 버텍스 그룹 가중치를 한 번의 순회로 numpy 배열({키: float32[n]})로 읽어옵니다.
    """
    import numpy as np
    n = len(obj.data.vertices)
    arrs = {k: np.zeros(n, dtype=np.float32) for k in group_map.values()}
    if not group_map:
        return arrs
    for v in obj.data.vertices:
        for g in v.groups:
            key = group_map.get(g.group)
            if key is not None:
                arrs[key][v.index] = g.weight
    return arrs


def world_vertex_positions(obj):
    """오브젝트의 모든 정점을 월드 좌표 (n, 3) float64 numpy 배열로 일괄 반환합니다."""
    import numpy as np
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    if n:
        obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    m = np.array(obj.matrix_world, dtype=np.float64)
    return co @ m[:3, :3].T + m[:3, 3]


def mesh_edge_arrays(mesh):
    """메쉬 엣지의 양 끝 정점 인덱스를 (e0, e1) int64 numpy 배열로 반환합니다."""
    import numpy as np
    m = len(mesh.edges)
    if m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    arr = np.empty(m * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", arr)
    arr = arr.reshape(-1, 2)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)


def label_components(n, e0, e1):
    """
    정점 n개와 엣지 (e0, e1)로 이루어진 그래프의 연결 요소 라벨(각 요소의 대표 정점 인덱스)을 numpy로 계산합니다.
    파이썬 BFS보다 훨씬 빠르며 정점 수십만 개도 즉시 처리합니다.
    """
    import numpy as np
    lab = np.arange(n, dtype=np.int64)
    if len(e0) == 0:
        return lab
    for _ in range(128):
        la = lab[e0]
        lb = lab[e1]
        mn = np.minimum(la, lb)
        new = lab.copy()
        np.minimum.at(new, la, mn)
        np.minimum.at(new, lb, mn)
        while True:
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def get_local_axes(obj):
    """블렌더 월드 기준 Up(+Z), Right(+X), Forward(-Y)가 오브젝트 로컬 공간에서 어느 방향인지 반환합니다."""
    try:
        inv_rot = obj.matrix_world.to_3x3().inverted()
        up_vec = (inv_rot @ Vector((0.0, 0.0, 1.0))).normalized()
        right_vec = (inv_rot @ Vector((1.0, 0.0, 0.0))).normalized()
        fwd_vec = (inv_rot @ Vector((0.0, -1.0, 0.0))).normalized()
        if up_vec.length > 0.5 and right_vec.length > 0.5:
            return up_vec, right_vec, fwd_vec
    except Exception:
        pass
    return Vector((0.0, 0.0, 1.0)), Vector((1.0, 0.0, 0.0)), Vector((0.0, -1.0, 0.0))


def point_segment_dist_and_t(p, a, b):
    """점 p와 선분 a-b 사이의 최단 거리 및 선분 위 위치 비율 t (0.0=head, 1.0=tail)를 반환합니다."""
    ab = b - a
    len_sq = ab.length_squared
    if len_sq < 1e-8:
        return (p - a).length, 0.0
    t = (p - a).dot(ab) / len_sq
    t_clamped = max(0.0, min(1.0, t))
    proj = a + ab * t_clamped
    return (p - proj).length, t_clamped


def axis_angle_matrix(axis, angle):
    """단위 축 벡터 axis와 라디안 각도 angle에 대한 3x3 회전 행렬을 반환합니다."""
    if abs(angle) < 1e-6 or axis.length_squared < 1e-8:
        return Matrix.Identity(3)
    return Matrix.Rotation(angle, 3, axis)


def compute_density_invariant_geometric_center(pts, n_bins=16):
    """
    버텍스 밀도 편향(지퍼, 솔기, 세밀한 장신구 등 특정 부위의 정점 밀집)을 배제하고
    순수한 3D 공간 형태의 기하학적 중심(Bounding Extents & Uniform Voxel Centroid)을 계산합니다.
    반환값: (geom_center: Vector, bmin: Vector, bmax: Vector)
    """
    import numpy as np
    pts_arr = np.array(pts, dtype=np.float64)
    if len(pts_arr) == 0:
        return Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0)), Vector((0.0, 0.0, 0.0))

    # 1. 이상치 제거된 3D Bounding Box (2% ~ 98% 백분위수)
    p2 = np.percentile(pts_arr, 2, axis=0)
    p98 = np.percentile(pts_arr, 98, axis=0)
    bmin = Vector((float(p2[0]), float(p2[1]), float(p2[2])))
    bmax = Vector((float(p98[0]), float(p98[1]), float(p98[2])))

    # 2. 균일 3D 복셀 그리드 점유도(Occupancy) 계산 -> 정점 수 편향 완전 배제
    spans = np.maximum(p98 - p2, 1e-4)
    coords = np.clip(np.floor((pts_arr - p2) / spans * n_bins).astype(np.int32), 0, n_bins - 1)
    unique_voxels = np.unique(coords, axis=0)

    voxel_centers = p2 + (unique_voxels + 0.5) * (spans / n_bins)
    geom_center_np = np.mean(voxel_centers, axis=0)
    box_center_np = (p2 + p98) * 0.5

    # 바운딩 박스 중심과 복셀 중심의 블렌딩 (형태의 전체적인 가운데 지향)
    final_np = 0.5 * geom_center_np + 0.5 * box_center_np
    geom_center = Vector((float(final_np[0]), float(final_np[1]), float(final_np[2])))

    return geom_center, bmin, bmax


def draw_help_button(layout, text):
    """패널 우측 상단에 공통 도움말(?) 아이콘 버튼을 그립니다."""
    row = layout.row()
    row.alignment = 'RIGHT'
    op = row.operator("object.xxmi_phys_help_tooltip", text="", icon='QUESTION', emboss=False)
    op.text = text


class XXMI_PHYS_OT_help_tooltip(bpy.types.Operator):
    bl_idname = "object.xxmi_phys_help_tooltip"
    bl_label = "도움말"

    text: bpy.props.StringProperty()

    @classmethod
    def description(cls, context, properties):
        return properties.text if properties else ""

    def execute(self, context):
        return {'FINISHED'}


classes = (
    XXMI_PHYS_OT_help_tooltip,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
