import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_rigid(bpy.types.Panel):
    """XXMI Custom Physics - 2. 강체 설정 패널"""
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_rigid"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'XXMI Physics'
    bl_order = 1
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.label(text="2. 강체 설정", icon='PINNED')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        # 도움말 툴팁 버튼
        draw_help_button(
            layout,
            (
                "[2단계: 강체 설정 가이드]\n\n"
                "• 물리 파츠 위에 부착된 단추, 배지, 리본 끈, 브로치 등은 모핑 동작 시 형태를 유지(강체)해야 합니다.\n"
                "• [+] 버튼으로 강체 슬롯을 추가하고 [선택 정점 할당]으로 부품 정점들을 등록하세요.\n"
                "• [표면 간접 추종]: 모든 강체는 맞닿은 메쉬 표면의 움직임을 자동으로 이어받아 형태를 유지합니다.\n"
                "  - 단추/배지 같은 단일 부품은 위치 표면에 맞춰 고정되어 움직입니다.\n"
                "  - 가슴 앞 리본 끈처럼 여러 본에 걸친 장식은 양쪽 표면의 움직임을 이어받아 찢어짐이나 이탈 없이 자연스럽게 따라 움직입니다!\n"
                "• [간접 영향력 배율]: 표면 대비 강체의 흔들림 전달 크기입니다 (기본값: 1.0)."
            ),
        )

        # 1. 다중 강체 슬롯 목록 (UIList)
        row = layout.row()
        row.template_list(
            "XXMI_UL_rigid_items", "",
            props, "rigid_items",
            props, "rigid_item_index",
            rows=3
        )
        col_btns = row.column(align=True)
        col_btns.operator("object.xxmi_rigid_item_add", text="", icon='ADD')
        col_btns.operator("object.xxmi_rigid_item_remove", text="", icon='REMOVE')

        # 2. 선택된 슬롯 상세 설정
        if 0 <= props.rigid_item_index < len(props.rigid_items):
            item = props.rigid_items[props.rigid_item_index]
            box_item = layout.box()
            box_item.use_property_split = True
            box_item.use_property_decorate = False

            box_item.prop(item, "name", text="부품 이름")
            box_item.prop(item, "weight_scale", text="간접 영향력 배율")

            row_acts = box_item.row(align=True)
            row_acts.operator("object.xxmi_rigid_item_assign", text="선택 정점 재할당", icon='FILE_REFRESH')
            row_acts.operator("object.xxmi_rigid_item_select_verts", text="확인", icon='RESTRICT_SELECT_OFF')
        else:
            layout.label(text="등록된 강체가 없습니다. [+]를 눌러 추가하세요.", icon='INFO')

        layout.separator()

        # 모핑 본에서 강체 정점 자동 제외 및 가중치 갱신 편의 버튼
        box_quick = layout.box()
        col_q = box_quick.column(align=True)
        col_q.operator(
            "object.xxmi_subtract_rigid_from_all_bones",
            text="모핑 본에서 강체 정점 자동 제외",
            icon='SELECT_SUBTRACT'
        )
        col_q.operator("object.xxmi_rebake_breast_weights", text="강체 가중치 즉시 갱신", icon='FILE_REFRESH')


classes = (
    XXMI_PHYS_PT_rigid,
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
