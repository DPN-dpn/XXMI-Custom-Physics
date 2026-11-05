import bpy


BONE_TYPE_ICONS = {
    'MORPH': 'SHAPEKEY_DATA',
    'TAIL': 'BONE_DATA',
    'HAIR': 'STRANDS',
    'SKIRT': 'MESH_GRID',
    'WING': 'OUTLINER_OB_ARMATURE',
}

BONE_TYPE_LABELS = {
    'MORPH': '모핑',
    'TAIL': '꼬리',
    'HAIR': '헤어',
    'SKIRT': '스커트',
    'WING': '날개',
}


class XXMI_UL_bone_items(bpy.types.UIList):
    """물리 본 슬롯 목록을 그리는 UIList입니다."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        btype = getattr(item, "bone_type", 'MORPH')
        bicon = BONE_TYPE_ICONS.get(btype, 'BONE_DATA')
        blabel = BONE_TYPE_LABELS.get(btype, '물리')

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text="", icon=bicon)
            row.prop(item, "name", text="", emboss=False)
            sub = row.row()
            sub.alignment = 'RIGHT'
            sub.label(text=f"[{blabel} | {item.vert_count}개 | W:{item.peak_weight:.2f}]")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon=bicon)


classes = (
    XXMI_UL_bone_items,
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
