import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_injector(bpy.types.Panel):
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_injector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XXMI Physics"
    bl_order = 3
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="4. 모드에 물리 주입", icon='SHADERFX')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        draw_help_button(
            layout,
            (
                "[4단계: 모드에 물리 주입 & 롤백]\n\n"
                "1. [대상 게임]을 선택합니다.\n"
                "2. 물리를 적용할 [모드 폴더]를 지정합니다.\n"
                "3. [모드에 물리 주입]을 누르면 인게임에서 동작하는 물리 효과가 모드에 적용됩니다.\n\n"
                "• [모드 폴더에서 물리 설정 불러오기]: 이미 물리가 주입된 모드 폴더에서 본 위치와 물리 파라미터를 다시 블렌더로 가져옵니다.\n"
                "• [인게임 ON/OFF 토글 사용]: 게임 실행 중 단축키(기본: h)를 눌러 물리 적용 전/후를 실시간으로 비교할 수 있습니다.\n"
                "• [물리 효과 롤백 (원본 복원)]: 주입된 물리 효과를 완전히 제거하고 모드를 원본 상태로 복원합니다."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(align=True)
        col.prop(props, "target_game")
        col.separator()
        row = col.row(align=True)
        row.prop(props, "target_mod_dir", text="모드 폴더")
        row.operator("object.xxmi_physics_dir_picker", text="", icon='FILE_FOLDER')

        col.separator()
        col.prop(props, "debug_mode")
        if props.debug_mode:
            box = layout.box()
            bcol = box.column(align=True)
            bcol.prop(props, "debug_toggle_key")

        layout.separator()

        col_mod_actions = layout.column(align=True)
        col_mod_actions.enabled = bool(props.target_mod_dir and props.target_mod_dir.strip())
        col_mod_actions.operator("object.xxmi_apply_custom_physics", text="모드에 물리 주입", icon='SHADERFX')
        col_mod_actions.operator("object.xxmi_load_physics_from_mod", text="모드 폴더에서 물리 설정 불러오기", icon='IMPORT')
        col_mod_actions.operator("object.xxmi_rollback_custom_physics", text="물리 효과 롤백 (원본 복원)", icon='RECOVER_LAST')


classes = (
    XXMI_PHYS_PT_injector,
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
