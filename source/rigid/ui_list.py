import bpy


class XXMI_UL_rigid_items(bpy.types.UIList):
    """강체 부품 목록을 그리는 UIList입니다."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text="", icon='PINNED')
            row.prop(item, "name", text="", emboss=False)
            bone_label = "자동"
            if item.target_bone == 'BREAST_L':
                bone_label = "Breast.L"
            elif item.target_bone == 'BREAST_R':
                bone_label = "Breast.R"
            elif item.target_bone == 'CENTER':
                bone_label = "중앙일체화"
            elif item.target_bone != 'AUTO':
                bone_label = item.target_bone
            sub = row.row()
            sub.alignment = 'RIGHT'
            sub.label(text=f"[{bone_label} | {item.vert_count}개]")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='PINNED')


classes = (
    XXMI_UL_rigid_items,
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
