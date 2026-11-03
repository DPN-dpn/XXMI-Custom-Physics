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
        layout.label(text="1. 가슴 선택 & 본 생성", icon='SPHERE')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        draw_help_button(
            layout,
            (
                "[1단계: 가슴 선택 & 물리 본 생성]\n\n"
                "1. 에디트 모드에서 가슴 표면을 선택하고 [1. 선택 영역을 가슴으로 등록]을 누릅니다.\n"
                "2. 단추, 브로치, 장식 등 모핑 시 일그러지면 안 되는 부위가 있다면 에디트 모드에서 선택 후 [2. 선택 부위를 강체로 등록]을 누릅니다. (강체가 없다면 바로 3단계로 이동)\n"
                "3. [3. 가슴 물리 본 생성 (강체 제외)]를 누릅니다.\n"
                "4. 뷰포트에서 생성된 본의 위치나 각도를 확인하고, 필요 시 조정한 뒤 [4. 본 기준 가중치(웨이트) 자동 계산]을 누릅니다.\n"
                "5. '2. 물리 설정 & 프리뷰'에서 움직임을 테스트하고 '3. 모드에 물리 주입'으로 인게임에 적용합니다.\n\n"
                "※ 하단의 [세부 설정 및 관리 도구]를 펼치면 가중치 퍼짐 조절, 정점 다시 선택, 초기화 기능을 사용할 수 있습니다."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        # --- 메인 작업 단계 (1 -> 2 -> 3 -> 4) ---
        box_main = layout.box()
        b_col = box_main.column(align=True)
        b_col.label(text="가슴 물리 설정 순서:", icon='SETTINGS')
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_register_breast_selection",
            text="1. 선택 영역을 가슴으로 등록",
            icon='SPHERE'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_mark_selection_rigid",
            text="2. 선택 부위를 강체로 등록",
            icon='RADIOBUT_ON'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_generate_breast_rig",
            text="3. 가슴 물리 본 생성 (강체 제외)",
            icon='BONE_DATA'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_rebake_breast_weights",
            text="4. 본 기준 가중치(웨이트) 자동 계산",
            icon='MOD_VERTEX_WEIGHT'
        )

        layout.separator()

        # --- 통합 접이식 세부 설정 & 관리 도구 (아코디언 UI) ---
        box_adv = layout.box()

        # 헤더 토글 버튼: 전체 행 클릭 상호작용 및 라벨 잘림 방지 (use_property_split = False)
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
            col_adv.prop(props, "falloff_power")
            col_adv.prop(props, "lock_registered_verts")

            col_adv.separator()

            # 2) 정점 관리 도구 (버튼들은 풀 너비로 깔끔하게 배치)
            col_tools = box_adv.column(align=True)
            col_tools.use_property_split = False
            col_tools.use_property_decorate = False
            col_tools.label(text="정점 관리 도구:", icon='RESTRICT_SELECT_OFF')
            col_tools.operator(
                "object.xxmi_select_breast_vertices",
                text="가슴 정점 다시 선택",
                icon='RESTRICT_SELECT_OFF'
            )
            col_tools.operator(
                "object.xxmi_select_rigid_vertices",
                text="강체 정점 다시 선택",
                icon='RESTRICT_SELECT_OFF'
            )
            col_tools.operator(
                "object.xxmi_unmark_selection_rigid",
                text="선택 부위 강체 해제",
                icon='X'
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
