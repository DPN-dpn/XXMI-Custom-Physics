import bpy
from ..utils import draw_help_button


class XXMI_PHYS_PT_simulation(bpy.types.Panel):
    """XXMI Custom Physics - 3. 물리 파라미터 & 프리뷰 패널"""
    bl_label = ""
    bl_idname = "XXMI_PHYS_PT_simulation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XXMI Physics"
    bl_order = 2
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        layout = self.layout
        layout.label(text="3. 물리 파라미터 & 프리뷰", icon='PHYSICS')

    def draw(self, context):
        layout = self.layout
        props = context.scene.xxmi_custom_physics_props

        draw_help_button(
            layout,
            (
                "[3단계: 가중치 계산 & 물리 프리뷰 가이드]\n\n"
                "• 1단계(본)와 2단계(강체) 설정이 완료된 후, 각 본의 가중치와 물리 파라미터를 조절합니다.\n"
                "• 각 본마다 피크 가중치, 감쇠, 부드러움과 흔들림 크기, 탄력 속도, 감쇠, 관성을 독립적으로 설정할 수 있습니다.\n"
                "• [본 & 강체 가중치 자동 계산 & 블렌딩]을 누르면 본들의 거리 감쇠 가중치와 강체 고정 가중치가 한 번에 계산·적용됩니다.\n"
                "• [뷰포트 실시간 물리 프리뷰 시작]을 켜고 슬라이더를 조절하여 움직임을 즉시 확인해보세요."
            ),
        )

        layout.use_property_split = True
        layout.use_property_decorate = False

        # 다중 물리 본 선택 UIList
        if len(props.bone_items) > 0:
            layout.label(text="조절 대상 물리 본:", icon='ARMATURE_DATA')
            row = layout.row()
            row.template_list(
                "XXMI_UL_bone_items", "sim",
                props, "bone_items",
                props, "bone_item_index",
                rows=2
            )

        target_item = None
        if 0 <= props.bone_item_index < len(props.bone_items):
            target_item = props.bone_items[props.bone_item_index]

        # 1. 가중치(웨이트) 파라미터 섹션
        box_w = layout.box()
        w_col = box_w.column(align=True)
        if target_item:
            w_col.label(text=f"'{target_item.name}' 가중치(웨이트) 설정:", icon='WPAINT_HLT')
            w_col.prop(target_item, "peak_weight")
            w_col.prop(target_item, "falloff_power")
            w_col.prop(target_item, "smooth_iterations")
            w_col.prop(target_item, "lock_registered_verts")
        else:
            w_col.label(text="전역 가중치(웨이트) 설정:", icon='WPAINT_HLT')
            w_col.prop(props, "peak_weight")
            w_col.prop(props, "falloff_power")
            w_col.prop(props, "smooth_iterations")
            w_col.prop(props, "lock_registered_verts")

        # 2. 물리 움직임 파라미터 섹션
        box_p = layout.box()
        p_col = box_p.column(align=True)
        if target_item:
            p_col.label(text=f"'{target_item.name}' 물리 거동 파라미터:", icon='PREFERENCES')
            p_col.prop(target_item, "strength")
            p_col.prop(target_item, "speed")
            p_col.prop(target_item, "damping")
            p_col.prop(target_item, "inertia")
            p_col.prop(target_item, "idle_motion")
            if target_item.bone_type in {'TAIL', 'HAIR', 'SKIRT', 'WING'}:
                p_col.separator()
                p_col.label(text=f"'{target_item.name}' 체인 관절 파라미터:", icon='CONSTRAINT_BONE')
                p_col.prop(target_item, "gravity_factor")
                p_col.prop(target_item, "stiffness")
            if len(props.bone_items) > 1:
                p_col.separator()
                p_col.operator("object.xxmi_copy_physics_params_to_all", text="모든 본에 현재 설정 일괄 복사", icon='COPYDOWN')
        else:
            p_col.label(text="전역 물리 움직임 파라미터:", icon='PREFERENCES')
            p_col.prop(props, "strength")
            p_col.prop(props, "speed")
            p_col.prop(props, "damping")
            p_col.prop(props, "inertia")
            p_col.prop(props, "idle_motion")

        p_col.separator()
        p_col.operator("object.xxmi_reset_physics_params", text="기본값으로 초기화", icon='LOOP_BACK')

        layout.separator()

        # 3. 핵심 실행 버튼 (가중치 계산 및 프리뷰)
        box_exec = layout.box()
        col_exec = box_exec.column(align=True)
        col_exec.label(text="가중치 계산 & 프리뷰 실행:", icon='PLAY')

        col_exec.operator(
            "object.xxmi_rebake_breast_weights",
            text="본 & 강체 가중치 자동 계산 & 블렌딩",
            icon='MOD_VERTEX_WEIGHT'
        )

        col_exec.separator()

        preview_text = "물리 프리뷰 정지" if props.is_preview_running else "뷰포트 실시간 물리 프리뷰 시작"
        preview_icon = 'PAUSE' if props.is_preview_running else 'PLAY'
        col_exec.operator(
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
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
