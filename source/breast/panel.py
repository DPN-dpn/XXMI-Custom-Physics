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
        layout.label(text="1. 가슴 선택 & 앵커 생성", icon='SPHERE')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        draw_help_button(
            layout,
            (
                "[가슴 물리 작업 순서]\n\n"
                "1. 에디트 모드에서 가슴 정점을 선택하고 [1. 선택 영역을 가슴 정점으로 등록]을 누릅니다.\n"
                "2. 단추, 브로치, 엠블럼 등 모핑되면 안 되는 단단한 장신구가 있다면 에디트 모드에서 선택 후 [2. 선택 부위를 강체 장신구로 등록]을 누릅니다. (장신구가 없다면 3단계로 바로 이동)\n"
                "3. [3. 가슴 앵커 본 생성 (강체 제외)]를 누릅니다.\n"
                "   ➔ 장신구를 제외한 순수 가슴 메쉬만을 기준으로 중심점과 돌출 방향(Nipple)을 평가하므로, 장신구 쪽으로 본이 쏠리지 않고 정확하게 생성됩니다.\n"
                "4. 뷰포트에서 생성된 본(XXMI_Breast_Rig)의 위치/각도를 확인하고 필요 시 미세 조정한 뒤, [4. 본 기준 가중치 계산 & 베이크]를 누릅니다.\n"
                "5. '2. 가슴 물리 파라미터 & 프리뷰'에서 움직임을 테스트하고 '3. 모드에 바스트모핑 주입'으로 인게임에 적용합니다."
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
            text="1. 선택 영역을 가슴 정점으로 등록",
            icon='SPHERE'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_mark_selection_rigid",
            text="2. 선택 부위를 강체 장신구로 등록",
            icon='RADIOBUT_ON'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_generate_breast_rig",
            text="3. 가슴 앵커 본 생성 (강체 제외)",
            icon='BONE_DATA'
        )
        b_col.separator(factor=0.5)

        b_col.operator(
            "object.xxmi_rebake_breast_weights",
            text="4. 본 기준 가중치 계산 & 베이크",
            icon='MOD_VERTEX_WEIGHT'
        )

        layout.separator()

        # --- 정점 재선택 & 강체 관리 도구 ---
        box_tools = layout.box()
        t_col = box_tools.column(align=True)
        t_col.label(text="정점 재선택 & 강체 관리 도구:", icon='RESTRICT_SELECT_OFF')
        t_col.operator(
            "object.xxmi_select_breast_vertices",
            text="가슴 정점 재선택",
            icon='RESTRICT_SELECT_OFF'
        )
        t_col.operator(
            "object.xxmi_select_rigid_vertices",
            text="강체 장신구 정점 재선택",
            icon='RESTRICT_SELECT_OFF'
        )
        t_col.operator(
            "object.xxmi_unmark_selection_rigid",
            text="선택 부위 강체 등록 해제",
            icon='X'
        )

        layout.separator()

        # --- 가중치 세부 옵션 ---
        box_opt = layout.box()
        o_col = box_opt.column(align=True)
        o_col.label(text="가중치 감쇠 설정:", icon='MOD_VERTEX_WEIGHT')
        o_col.prop(props, "falloff_power")
        o_col.prop(props, "lock_registered_verts")

        layout.separator()

        # --- 초기화 ---
        box_sub = layout.box()
        s_col = box_sub.column(align=True)
        s_col.label(text="초기화:", icon='TRASH')
        s_col.operator(
            "object.xxmi_clear_breast_physics",
            text="가슴 물리 설정 초기화",
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
