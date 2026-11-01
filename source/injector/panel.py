import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_injector(bpy.types.Panel):
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_injector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XXMI Physics"
    bl_order = 2

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="3. 모드에 바스트모핑 주입", icon='SHADERFX')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        draw_help_button(
            layout,
            (
                "[3단계: 타겟 모드에 바스트모핑 HLSL 주입 & 롤백]\n\n"
                "• 익스포트된 타겟 모드의 폴더를 지정하고 [모드에 바스트모핑 주입]을 누르면\n"
                "  가슴 정점 가중치 버퍼(CustomPhysicsMap.buf)와 실시간 HLSL Compute Shader를 생성하고\n"
                "  모드의 .ini에 자동으로 연결합니다.\n\n"
                "• 디버그 모드: 인게임에서 h 키(또는 지정한 키)로 바스트모핑 ON/OFF를 즉시 비교할 수 있습니다.\n"
                "• 물리 효과 롤백: 주입된 모든 파일과 .ini 구문을 100% 원본 상태로 복원합니다."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(props, "target_mod_dir", text="모드 폴더")
        row.operator("object.xxmi_physics_dir_picker", text="", icon='FILE_FOLDER')

        col.separator()
        col.operator("object.xxmi_load_physics_from_mod", text="모드 폴더에서 모핑 정보 불러오기", icon='IMPORT')

        col.separator()
        col.prop(props, "debug_mode")
        if props.debug_mode:
            box = layout.box()
            bcol = box.column(align=True)
            bcol.prop(props, "debug_toggle_key")

        layout.separator()
        layout.operator("object.xxmi_apply_custom_physics", text="모드에 바스트모핑 주입", icon='SHADERFX')
        layout.operator("object.xxmi_rollback_custom_physics", text="바스트모핑 효과 롤백", icon='RECOVER_LAST')


classes = (
    XXMI_PHYS_PT_injector,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
