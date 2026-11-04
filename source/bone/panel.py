import bpy


class XXMI_PHYS_PT_bone(bpy.types.Panel):
    """XXMI Custom Physics - 1. 물리 본 & 웨이트 설정 패널"""
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_bone"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'XXMI Physics'
    bl_order = 0

    def draw_header(self, context):
        self.layout.label(text="1. 물리 본 & 웨이트 설정", icon='BONE_DATA')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        # 도움말 툴팁 버튼
        row_top = layout.row(align=True)
        sub_r = row_top.row()
        sub_r.alignment = 'RIGHT'
        op_h = sub_r.operator("object.xxmi_phys_help_tooltip", text="", icon='QUESTION')
        op_h.tooltip_title = "물리 본 & 웨이트 설정 가이드"
        op_h.tooltip_message = (
            "캐릭터에 물리 본을 추가하고 웨이트를 자동 생성하는 단계입니다.\n\n"
            "1. 원하는 [본 종류](가슴, 향후 꼬리/머리카락 등)를 선택합니다.\n"
            "2. 에디트 모드에서 물리 본을 적용할 표면 정점들을 선택하고 [1. 가슴 영역 선택 등록]을 누릅니다.\n"
            "3. [2. 가슴 물리 본 생성]을 누르면 해부학적 위치에 알맞은 물리 본(Armature)이 자동 배치됩니다.\n"
            "4. 뷰포트에서 본의 위치/각도/반경을 자유롭게 다듬은 후 [3. 본 기준 가중치(웨이트) 자동 계산]을 누르면 "
            "매끄러운 LBS 가중치가 메쉬에 베이킹됩니다."
        )

        # 0. 본 종류 선택
        box_type = layout.box()
        box_type.use_property_split = True
        box_type.use_property_decorate = False
        box_type.prop(props, "bone_type", text="본 종류", icon='BONE_DATA')

        # 1. 메인 작업 흐름 (본 중심 3단계)
        col_main = layout.column(align=True)
        col_main.operator("object.xxmi_register_breast_selection", text="1. 가슴 영역 선택 등록", icon='MESH_DATA')
        col_main.operator("object.xxmi_generate_breast_rig", text="2. 가슴 물리 본 생성", icon='ARMATURE_DATA')
        col_main.operator("object.xxmi_rebake_breast_weights", text="3. 본 기준 가중치 자동 계산", icon='MOD_VERTEX_WEIGHT')

        # 2. 고급 웨이트 세부 조정 (접이식)
        box_adv = layout.box()
        row_adv = box_adv.row()
        row_adv.prop(
            props, "ui_show_advanced_breast",
            icon="TRIA_DOWN" if getattr(props, "ui_show_advanced_breast", False) else "TRIA_RIGHT",
            text="고급 가중치 설정",
            emboss=False
        )
        if getattr(props, "ui_show_advanced_breast", False):
            col_adv = box_adv.column(align=True)
            col_adv.use_property_split = True
            col_adv.use_property_decorate = False
            col_adv.prop(props, "falloff_power", text="가중치 감쇠(Falloff)")
            col_adv.prop(props, "peak_weight", text="최대 가중치(Peak)")
            col_adv.prop(props, "smooth_iterations", text="스무딩 반복 수")
            col_adv.prop(props, "lock_registered_verts", text="등록 정점만 가중치 적용")

        # 3. 보조 도구 (정점 다시 선택 및 전체 초기화)
        box_tools = layout.box()
        col_t = box_tools.column(align=True)
        col_t.operator("object.xxmi_select_breast_vertices", text="가슴 정점 다시 선택", icon='RESTRICT_SELECT_OFF')
        col_t.operator("object.xxmi_clear_breast_physics", text="가슴 물리 전체 초기화", icon='TRASH')


classes = (
    XXMI_PHYS_PT_bone,
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
