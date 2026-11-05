import bpy


class XXMI_UL_rigid_items(bpy.types.UIList):
    """강체 부품 목록을 그리는 UIList입니다."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text="", icon='PINNED')
            row.prop(item, "name", text="", emboss=False)
            sub = row.row()
            sub.alignment = 'RIGHT'
            sub.label(text=f"[{item.vert_count}개 | x{item.weight_scale:.2f}]")
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
