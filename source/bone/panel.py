import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_bone(bpy.types.Panel):
    """XXMI Custom Physics - 1. 물리 본 & 웨이트 설정 패널"""
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_bone"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'XXMI Physics'
    bl_order = 0

    def draw_header(self, context):
        self.layout.label(text="1. 모핑(물리) 본 & 웨이트 설정", icon='BONE_DATA')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        # 도움말 툴팁 버튼
        draw_help_button(
            layout,
            (
                "[1단계: 모핑 본 & 웨이트 설정 가이드]\n\n"
                "• 가슴, 엉덩이, 복부 등 모든 모핑/물리 본을 슬롯 목록으로 독립 관리합니다.\n"
                "• 좌우 고정 없이 한쪽씩 모핑 본을 지정해도 본들이 알아서 블렌드되어 적용됩니다.\n"
                "• 특정 부위에 버텍스 밀도가 몰려있어도 전체적인 3D 공간 형태의 중심을 계산하여 본을 배치합니다.\n"
                "• [3. 본 기준 가중치 자동 계산 & 블렌딩]을 누르면 각 본의 영역별 가중치가 계산되고 겹치는 부위는 자연스럽게 블렌딩됩니다."
            ),
        )

        # 1. 다중 모핑 본 슬롯 목록 (UIList)
        layout.label(text="모핑 본 슬롯 목록:", icon='ARMATURE_DATA')
        row = layout.row()
        row.template_list(
            "XXMI_UL_bone_items", "",
            props, "bone_items",
            props, "bone_item_index",
            rows=3
        )
        col_btns = row.column(align=True)
        col_btns.operator("object.xxmi_bone_item_add", text="", icon='ADD')
        col_btns.operator("object.xxmi_bone_item_remove", text="", icon='REMOVE')

        # 2. 선택된 본 슬롯 상세 설정
        if 0 <= props.bone_item_index < len(props.bone_items):
            item = props.bone_items[props.bone_item_index]
            box_item = layout.box()
            box_item.use_property_split = True
            box_item.use_property_decorate = False

            box_item.prop(item, "name", text="본 이름")
            box_item.prop(item, "bone_type", text="본 종류")

            row_acts = box_item.row(align=True)
            row_acts.operator("object.xxmi_bone_item_assign", text="선택 정점 할당", icon='PLUS')
            row_acts.operator("object.xxmi_bone_item_select_verts", text="확인", icon='RESTRICT_SELECT_OFF')
            row_acts.operator("object.xxmi_bone_item_remove_verts", text="제외", icon='X')

            # 개별 본 가중치 설정
            col_w = box_item.column(align=True)
            col_w.separator()
            col_w.label(text=f"'{item.name}' 가중치(웨이트) 설정:", icon='MOD_VERTEX_WEIGHT')
            col_w.prop(item, "peak_weight", text="최대 가중치(Peak)")
            col_w.prop(item, "falloff_power", text="가중치 퍼짐(Falloff)")
            col_w.prop(item, "smooth_iterations", text="스무딩 반복 수")
            col_w.prop(item, "lock_registered_verts", text="등록 영역만 유지")

            if len(props.bone_items) > 1:
                box_item.separator()
                box_item.operator("object.xxmi_copy_bone_weights_to_all", text="가중치 설정 다른 본에 복사", icon='COPYDOWN')
        else:
            layout.label(text="등록된 본이 없습니다. [1. 모핑 영역 선택 등록] 또는 [+]를 누르세요.", icon='INFO')

        layout.separator()

        # 3. 메인 작업 흐름 (본 중심 3단계)
        layout.label(text="모핑 본 작업 워크플로우:", icon='MOD_ARMATURE')
        col_main = layout.column(align=True)
        col_main.operator("object.xxmi_register_breast_selection", text="1. 모핑 영역 선택 등록", icon='MESH_DATA')
        col_main.operator("object.xxmi_generate_breast_rig", text="2. 모핑 본 자동 생성", icon='ARMATURE_DATA')
        col_main.operator("object.xxmi_rebake_breast_weights", text="3. 본 기준 가중치 자동 계산 & 블렌딩", icon='MOD_VERTEX_WEIGHT')

        # 4. 보조 도구 (전체 정점 다시 선택 및 전체 초기화)
        box_tools = layout.box()
        col_t = box_tools.column(align=True)
        col_t.operator("object.xxmi_select_breast_vertices", text="모핑 정점 전체 다시 선택", icon='RESTRICT_SELECT_OFF')
        col_t.operator("object.xxmi_clear_breast_physics", text="모핑 본 & 물리 전체 초기화", icon='TRASH')


classes = (
    XXMI_PHYS_PT_bone,
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
