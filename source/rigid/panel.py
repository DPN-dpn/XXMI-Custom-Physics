import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_rigid(bpy.types.Panel):
    """XXMI Custom Physics - 2. 강체(액세서리) 설정 패널"""
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_rigid"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'XXMI Physics'
    bl_order = 1

    def draw_header(self, context):
        self.layout.label(text="2. 강체(액세서리) 설정", icon='PINNED')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        # 도움말 툴팁 버튼
        draw_help_button(
            layout,
            (
                "[2단계: 강체(액세서리) 설정 가이드]\n\n"
                "• 모핑 파츠(가슴, 의상, 복부 등) 위에 부착된 단추, 무전기, 리본, 브로치 등은 모핑 동작 시 형태를 유지(강체)해야 합니다.\n"
                "• [+] 버튼으로 강체 슬롯을 추가합니다.\n"
                "• 에디트 모드에서 부품 정점들을 선택하고 [선택 정점 할당]을 누릅니다. (여러 오브젝트에 걸친 부품도 동시 지원)\n"
                "• [목표 본]을 지정하면 해당 모핑 본의 움직임과 100% 동일하게 고정되어 따라 움직입니다.\n"
                "• 설정 후 바로 [강체 가중치 즉시 갱신]을 누르거나, [3. 가중치 계산 & 프리뷰] 패널로 이동하여 물리 파라미터와 함께 일괄 계산할 수 있습니다."
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
            box_item.prop(item, "target_bone", text="목표 본")
            box_item.prop(item, "weight_scale", text="움직임 배율")

            row_acts = box_item.row(align=True)
            row_acts.operator("object.xxmi_rigid_item_assign", text="선택 정점 할당", icon='PLUS')
            row_acts.operator("object.xxmi_rigid_item_select_verts", text="확인", icon='RESTRICT_SELECT_OFF')
            row_acts.operator("object.xxmi_rigid_item_remove_verts", text="제외", icon='X')
        else:
            layout.label(text="등록된 강체가 없습니다. [+]를 눌러 추가하세요.", icon='INFO')

        layout.separator()

        # 강체 가중치 즉시 갱신 편의 버튼
        box_quick = layout.box()
        col_q = box_quick.column(align=True)
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
