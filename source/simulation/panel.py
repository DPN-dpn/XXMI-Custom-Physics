import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_simulation(bpy.types.Panel):
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_simulation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XXMI Physics"
    bl_order = 1

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="2. 물리 설정 & 프리뷰", icon='PHYSICS')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        draw_help_button(
            layout,
            (
                "[2단계: 물리 움직임 설정 & 프리뷰]\n\n"
                "• 캐릭터가 움직일 때 가슴이 반응하는 흔들림 크기, 튀는 속도, 멈추는 탄성을 조절합니다.\n"
                "• [뷰포트 실시간 물리 프리뷰 시작]을 누르면 블렌더 화면에서 가슴이 통통 튀는 움직임을 실시간으로 확인할 수 있습니다.\n"
                "• 프리뷰를 켜둔 상태로 슬라이더를 움직여 마음에 드는 움직임으로 맞춰보세요."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        box_params = layout.box()
        p_col = box_params.column(align=True)
        p_col.label(text="물리 움직임 파라미터:", icon='PREFERENCES')
        p_col.prop(props, "strength")
        p_col.prop(props, "speed")
        p_col.prop(props, "damping")
        p_col.prop(props, "inertia")
        p_col.prop(props, "idle_motion")
        p_col.separator()
        p_col.operator("object.xxmi_reset_physics_params", text="기본값으로 초기화", icon='LOOP_BACK')

        layout.separator()

        box_prev = layout.box()
        pr_col = box_prev.column(align=True)
        pr_col.label(text="뷰포트 프리뷰:", icon='PLAY')

        preview_text = "물리 프리뷰 정지" if props.is_preview_running else "뷰포트 실시간 물리 프리뷰 시작"
        preview_icon = 'PAUSE' if props.is_preview_running else 'PLAY'
        pr_col.operator(
            "object.xxmi_preview_physics",
            text=preview_text,
            icon=preview_icon,
            depress=props.is_preview_running,
        )


classes = (
    XXMI_PHYS_PT_simulation,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
