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
        bpy.utils.unregister_class(cls)
