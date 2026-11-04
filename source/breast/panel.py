import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_breast(bpy.types.Panel):
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_breast"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XXMI Physics"
    bl_order = 0

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="1. 물리 본 & 강체 설정", icon='BONE_DATA')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        draw_help_button(
            layout,
            (
                "[1단계: 물리 본 & 다중 강체 설정]\n\n"
                "1. 에디트 모드에서 가슴 표면을 선택하고 [1. 선택 영역을 가슴으로 등록]을 누릅니다.\n"
                "2. 단추, 무전기, 리본, 브로치 등 찌그러지면 안 되는 장신구가 있다면 하단 [강체 관리자]에서 [+]로 강체 슬롯을 추가하고 [선택 정점 할당]을 누릅니다. (추종할 본도 지정 가능)\n"
                "3. [2. 가슴 물리 본 생성]을 누르면 뷰포트에 뼈대가 생성됩니다.\n"
                "4. 뷰포트에서 본의 위치나 각도를 필요 시 조정한 뒤 [3. 본 기준 가중치(웨이트) 자동 계산]을 누릅니다.\n"
                "5. '2. 시뮬레이션 & 프리뷰'에서 움직임을 테스트하고 '3. 모드에 물리 주입'으로 인게임에 적용합니다.\n\n"
                "※ 하단의 [세부 설정 및 관리 도구]를 펼치면 가중치 퍼짐 조절, 정점 다시 선택, 초기화 기능을 사용할 수 있습니다."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        # ── 1. 메인 물리 본 작업 단계 ──
        box_main = layout.box()
        b_col = box_main.column(align=True)
        b_col.label(text="물리 본 생성 순서:", icon='SETTINGS')
        b_col.separator(factor=0.5)

        b_col.prop(props, "bone_type", text="본 종류")
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_register_breast_selection",
            text="1. 선택 영역을 가슴으로 등록",
            icon='SPHERE'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_generate_breast_rig",
            text="2. 가슴 물리 본 생성",
            icon='BONE_DATA'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_rebake_breast_weights",
            text="3. 본 기준 가중치(웨이트) 자동 계산",
            icon='MOD_VERTEX_WEIGHT'
        )

        layout.separator(factor=0.7)

        # ── 2. 다중 강체 관리자 (Multi-Rigid Manager) ──
        box_rigid = layout.box()
        r_head = box_rigid.row()
        r_head.label(text=f"강체 부품 관리자 ({len(props.rigid_items)}개)", icon='SHAPEKEY_DATA')

        # UIList 및 +/- 버튼
        row_list = box_rigid.row()
        row_list.template_list(
            "XXMI_UL_rigid_items",
            "",
            props,
            "rigid_items",
            props,
            "rigid_item_index",
            rows=3
        )
        col_btns = row_list.column(align=True)
        col_btns.operator("object.xxmi_rigid_item_add", text="", icon='ADD')
        col_btns.operator("object.xxmi_rigid_item_remove", text="", icon='REMOVE')

        # 활성 강체 슬롯 세부 제어
        if 0 <= props.rigid_item_index < len(props.rigid_items):
            cur_item = props.rigid_items[props.rigid_item_index]
            box_detail = box_rigid.box()
            d_col = box_detail.column(align=True)
            d_col.use_property_split = True
            d_col.use_property_decorate = False

            d_col.prop(cur_item, "name", text="부품 이름")
            d_col.prop(cur_item, "target_bone", text="추종 본")
            d_col.separator(factor=0.5)

            # 정점 할당 / 선택 버튼 행
            r_acts = box_detail.row(align=True)
            r_acts.use_property_split = False
            r_acts.operator(
                "object.xxmi_rigid_item_assign",
                text="선택 정점 할당",
                icon='CHECKMARK'
            )
            r_acts.operator(
                "object.xxmi_rigid_item_select_verts",
                text="정점 선택",
                icon='RESTRICT_SELECT_OFF'
            )
            r_acts.operator(
                "object.xxmi_rigid_item_remove_verts",
                text="정점 제외",
                icon='X'
            )
        else:
            box_rigid.label(text="[+] 버튼을 눌러 새 강체 부품(단추, 무전기 등)을 추가하세요", icon='INFO')

        layout.separator(factor=0.7)

        # ── 3. 통합 접이식 세부 설정 & 관리 도구 (아코디언 UI) ──
        box_adv = layout.box()

        col_toggle = box_adv.column(align=True)
        col_toggle.use_property_split = False
        col_toggle.use_property_decorate = False
        adv_icon = 'TRIA_DOWN' if props.show_advanced_breast_settings else 'TRIA_RIGHT'
        col_toggle.prop(
            props,
            "show_advanced_breast_settings",
            text="세부 설정 및 관리 도구",
            icon=adv_icon,
            toggle=True
        )

        if props.show_advanced_breast_settings:
            col_adv = box_adv.column(align=True)
            col_adv.use_property_split = True
            col_adv.use_property_decorate = False
            col_adv.separator(factor=0.5)

            # 1) 가중치 세부 설정
            col_adv.label(text="가중치 세부 설정:", icon='MOD_VERTEX_WEIGHT')
            col_adv.prop(props, "peak_weight")
            col_adv.prop(props, "smooth_iterations")
            col_adv.prop(props, "falloff_power")
            col_adv.prop(props, "lock_registered_verts")

            col_adv.separator()

            # 2) 정점 관리 도구
            col_tools = box_adv.column(align=True)
            col_tools.use_property_split = False
            col_tools.use_property_decorate = False
            col_tools.label(text="가슴 정점 관리:", icon='RESTRICT_SELECT_OFF')
            col_tools.operator(
                "object.xxmi_select_breast_vertices",
                text="가슴 정점 다시 선택",
                icon='RESTRICT_SELECT_OFF'
            )

            col_tools.separator(factor=0.5)

            # 3) 초기화
            col_tools.operator(
                "object.xxmi_clear_breast_physics",
                text="가슴 물리 설정 전체 초기화",
                icon='TRASH'
            )


classes = (
    XXMI_PHYS_PT_breast,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
