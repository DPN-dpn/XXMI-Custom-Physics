import bpy
from bpy.types import Panel
from ..utils import draw_help_button


class XXMI_PHYS_PT_updater(Panel):
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_updater"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "XXMI Physics"
    bl_order = 9
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="업데이트", icon="FILE_REFRESH")

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        draw_help_button(
            layout,
            (
                "[애드온 업데이트]\n"
                "GitHub 공식 저장소(DPN-dpn/XXMI-Custom-Physics)에서 최신 릴리스 버전을 확인하고\n"
                "자동으로 다운로드 및 설치합니다."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        latest_version = scene.get("xxmi_phys_latest_version", "")
        current_version = scene.get("xxmi_phys_current_version", "")
        update_available = scene.get("xxmi_phys_update_available", False)
        show_restart = scene.get("xxmi_phys_show_restart", False)

        if not latest_version:
            update_label = "업데이트 체크 필요"
        elif not update_available:
            update_label = "현재 최신 버전입니다"
        else:
            update_label = f"업데이트: {current_version} → {latest_version}"

        layout.operator("xxmi_phys.check_update", text="업데이트 체크", icon="FILE_REFRESH")

        row = layout.row()
        if show_restart:
            row.operator("wm.quit_blender", text="블렌더 종료(애드온 재실행)", icon="CANCEL")
        else:
            row.enabled = bool(update_available)
            row.operator("xxmi_phys.do_update", text=update_label, icon="IMPORT")

        layout.operator("xxmi_phys.open_github", text="GitHub", icon="URL")


classes = (
    XXMI_PHYS_PT_updater,
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

