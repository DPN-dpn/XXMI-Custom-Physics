import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_bone(bpy.types.Panel):
    """XXMI Custom Physics - 1. 모핑 본(스켈레톤) 설정 패널"""
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_bone"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'XXMI Physics'
    bl_order = 0

    def draw_header(self, context):
        self.layout.label(text="1. 모핑 본(스켈레톤) 설정", icon='BONE_DATA')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        # 도움말 툴팁 버튼
        draw_help_button(
            layout,
            (
                "[1단계: 모핑 본(스켈레톤) 설정 가이드]\n\n"
                "• 가슴, 엉덩이, 복부 등 움직일 모핑 부위를 지정하고 가상 물리 본을 생성합니다.\n"
                "• [1. 모핑 영역 선택 등록] ➔ [2. 모핑 본 자동 생성] 순서로 진행합니다.\n"
                "• 특정 부위에 버텍스가 몰려있어도 3D 공간 형태의 중심을 계산하여 본이 배치됩니다.\n"
                "• 본 생성이 완료되면 [2. 강체(액세서리) 설정]으로 넘어가거나 바로 [3. 가중치 계산 & 프리뷰]로 이동하세요."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        # 1. 메인 작업 흐름 (본 생성)
        layout.label(text="모핑 본 생성 워크플로우:", icon='MOD_ARMATURE')
        col_main = layout.column(align=True)
        col_main.operator("object.xxmi_register_breast_selection", text="1. 모핑 영역 선택 등록", icon='MESH_DATA')
        col_main.operator("object.xxmi_generate_breast_rig", text="2. 모핑 본 자동 생성", icon='ARMATURE_DATA')

        layout.separator()

        # 2. 다중 모핑 본 슬롯 목록 (UIList)
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

        # 3. 선택된 본 슬롯 상세 설정
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
        else:
            layout.label(text="등록된 본이 없습니다. [1. 모핑 영역 선택 등록] 또는 [+]를 누르세요.", icon='INFO')

        layout.separator()

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
