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
                "[1단계: 가슴 선택 & 장신구 보존 앵커 등록]\n\n"
                "1. 모드 메쉬 오브젝트(예: Body, 의상 등)를 Shift+선택하고 Tab(에디트 모드)으로 들어갑니다.\n"
                "   ➔ 단일 메쉬뿐만 아니라 여러 개로 분리된 메쉬를 동시에 선택하여 한 번에 작업할 수 있습니다.\n"
                "2. 가슴 표면 정점들을 박스나 브러시로 쓱 드래그하여 선택합니다.\n"
                "3. [선택 영역을 가슴 물리로 등록] 버튼을 클릭합니다.\n"
                "   ➔ 중심선(X=0) 기준으로 좌/우 가슴 앵커가 자동 배치됩니다.\n"
                "   ➔ 단추, 브로치 등 독립된 장신구는 메쉬 섬(Island) 분석을 통해 강체로 자동 보호됩니다.\n"
                "4. 특수한 장신구는 에디트 모드에서 선택 후 [선택 부위를 강체 장신구로 지정]을 누르세요."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        box_main = layout.box()
        b_col = box_main.column(align=True)
        b_col.label(text="가슴 정점 등록 & 본 동기화:", icon='EDITMODE_HLT')
        b_col.operator(
            "object.xxmi_register_breast_selection",
            text="선택 영역을 가슴 물리로 등록",
            icon='SPHERE'
        )
        b_col.separator()
        b_col.operator(
            "object.xxmi_select_breast_vertices",
            text="적용된 가슴 정점 재선택",
            icon='RESTRICT_SELECT_OFF'
        )
        b_col.separator()
        b_col.operator(
            "object.xxmi_rebake_breast_weights",
            text="조정된 본 위치 기준으로 가중치 재계산",
            icon='BONE_DATA'
        )

        layout.separator()

        box_opt = layout.box()
        o_col = box_opt.column(align=True)
        o_col.label(text="장신구 보호 및 감쇠 설정:", icon='MOD_VERTEX_WEIGHT')
        o_col.prop(props, "auto_detect_rigid")
        if props.auto_detect_rigid:
            o_col.prop(props, "rigid_max_verts")
        o_col.prop(props, "falloff_power")

        layout.separator()

        box_sub = layout.box()
        s_col = box_sub.column(align=True)
        s_col.label(text="수동 강체 지정 & 초기화:", icon='RESTRICT_SELECT_OFF')
        s_col.operator(
            "object.xxmi_mark_selection_rigid",
            text="선택 부위를 강체 장신구로 지정",
            icon='RADIOBUT_ON'
        )
        s_col.separator()
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

