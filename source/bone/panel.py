import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_bone(bpy.types.Panel):
    """XXMI Custom Physics - 1. 물리 본 & 영역 설정 패널"""
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_bone"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'XXMI Physics'
    bl_order = 0

    def draw_header(self, context):
        self.layout.label(text="1. 물리 본 & 영역 설정", icon='BONE_DATA')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        # 도움말 툴팁 버튼
        draw_help_button(
            layout,
            (
                "[1단계: 물리 본 & 영역 설정 가이드]\n\n"
                "• 에디트 모드에서 물리(흔들림/모핑) 효과를 줄 부위의 정점들을 선택하고 [+]를 누르면 슬롯과 가상 물리 본이 즉시 생성됩니다.\n"
                "• 가슴 전체처럼 좌/우가 이어진 의상이나 신체 부위는 한 번에 전체를 선택한 뒤 [선택 영역 절반 나눠 본 2개 추가]를 누르면 좌/우 독립 본 2개로 깔끔하게 분할 생성됩니다.\n"
                "• '본 반경 (영역/겹침 크기)'를 조절하면 스켈레톤 본의 감지 반경과 두 본 사이의 중앙 겹침(Cross-Blending) 폭을 직관적으로 조절할 수 있습니다.\n"
                "• 단추, 브로치, 무전기 등 강체(액세서리)가 붙어있는 경우, 일단 통째로 포함해서 본 슬롯을 생성하세요.\n"
                "• 그 후 2패널에서 강체를 등록하고, [강체 정점 자동 제외 (경계 유지)]를 누르면 본 표면과 맞닿아 있는 접합 경계 버텍스는 유지하면서 돌출된 강체 정점만 깔끔하게 제외됩니다!\n"
                "• 본 배치가 완료되면 [2. 강체(액세서리) 설정]으로 넘어가거나 바로 [3. 물리 파라미터 & 프리뷰]로 이동하세요."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        # 1. 새 본 생성 옵션
        row_new = layout.row(align=True)
        row_new.prop(props, "bone_type", text="본 종류 선택")
        row_new.operator("object.xxmi_bone_item_add", text="새 본 슬롯 추가", icon='ADD')

        row_split = layout.row(align=True)
        row_split.operator(
            "object.xxmi_bone_item_add_split_lr",
            text="선택 영역 절반 나눠 본 2개 추가 (가슴 등)",
            icon='MOD_MIRROR'
        )

        layout.separator()

        # 2. 다중 물리 본 슬롯 목록 (UIList)
        layout.label(text="물리 본 슬롯 목록:", icon='ARMATURE_DATA')
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
            box_item.prop(item, "radius_scale", text="본 반경 (영역/겹침 크기)")
            box_item.prop(props, "show_envelope_gizmo", text="뷰포트 본 반경(원통) 표시")

            if item.bone_type in {'TAIL', 'HAIR', 'SKIRT', 'WING'}:
                box_chain = box_item.box()
                box_chain.label(text="다관절 체인(FK) 구조 설정:", icon='CONSTRAINT_BONE')
                box_chain.prop(item, "chain_segments", text="마디(관절) 수")
                box_chain.prop(item, "gravity_factor", text="중력 영향력")
                box_chain.prop(item, "stiffness", text="관절 뻣뻣함")

            row_acts = box_item.row(align=True)
            row_acts.operator("object.xxmi_bone_item_assign", text="선택 정점 할당", icon='PLUS')
            row_acts.operator("object.xxmi_bone_item_select_verts", text="확인", icon='RESTRICT_SELECT_OFF')
            row_acts.operator("object.xxmi_bone_item_remove_verts", text="제외", icon='X')

            # 모핑 슬롯에서 강체 정점 자동 제외 (경계 버텍스 유지)
            row_sub = box_item.row(align=True)
            row_sub.operator(
                "object.xxmi_bone_item_subtract_rigid_verts",
                text="강체 정점 자동 제외 (경계 유지)",
                icon='SELECT_SUBTRACT'
            )
        else:
            layout.label(text="등록된 본이 없습니다. 에디트 모드에서 정점 선택 후 [+]를 누르세요.", icon='INFO')

        layout.separator()

        # 4. 보조 도구 (전체 본 재배치 및 전체 초기화)
        box_tools = layout.box()
        col_t = box_tools.column(align=True)
        col_t.operator("object.xxmi_generate_breast_rig", text="모든 물리 본 위치/크기 일괄 재배치", icon='ARMATURE_DATA')
        col_t.operator("object.xxmi_clear_breast_physics", text="물리 본 & 가중치 전체 초기화", icon='TRASH')


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
