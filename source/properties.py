import bpy


class XXMI_PG_custom_physics_props(bpy.types.PropertyGroup):
    # ── 1. 대상 메쉬 및 앵커 ──
    target_obj: bpy.props.PointerProperty(
        name="대상 메쉬",
        type=bpy.types.Object,
        description="가슴 물리를 적용할 캐릭터 모드 메쉬 오브젝트입니다",
        poll=lambda self, obj: obj.type == 'MESH'
    )

    anchor_obj: bpy.props.PointerProperty(
        name="가슴 앵커",
        type=bpy.types.Object,
        description="가슴 물리 모션의 중심점 역할을 하는 앵커 오브젝트(XXMI_Breast_Rig)입니다",
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )

    # ── 2. 가슴 가중치 감쇠 옵션 ──
    falloff_power: bpy.props.FloatProperty(
        name="가슴 감쇠 부드러움",
        description="가슴 꼭지점에서 몸통 둘레로 갈수록 가중치가 0으로 감쇠하는 부드러움 곡선 지수입니다 (높을수록 외곽이 단단히 고정됨)",
        default=1.6,
        min=0.5,
        max=4.0,
        step=10,
        precision=2
    )

    lock_registered_verts: bpy.props.BoolProperty(
        name="등록 정점 범위 엄격 유지",
        description="가중치 재계산 시 본 반경으로 인해 주변 정점이 자동으로 추가되거나 기존 정점이 탈락하지 않고, 처음에 선택/등록된 정점 범위만을 100% 엄격히 유지합니다",
        default=True
    )

    # ── 3. 가슴 물리 거동 파라미터 ──
    strength: bpy.props.FloatProperty(
        name="흔들림 강도",
        description="가슴 물리 바운스의 전체 진폭 배율입니다 (0.0이면 정지, 1.0이 기본)",
        default=1.0,
        min=0.0,
        max=3.0,
        step=5,
        precision=2
    )

    speed: bpy.props.FloatProperty(
        name="탄력 속도",
        description="가슴이 통통 튀는 복원 탄력 속도입니다",
        default=1.1,
        min=0.2,
        max=3.0,
        step=5,
        precision=2
    )

    damping: bpy.props.FloatProperty(
        name="탄성 감쇠",
        description="흔들린 후 원래 자리로 가라앉는 감쇠율입니다 (낮을수록 여운이 길고 출렁거림, 높을수록 단단히 빨리 멈춤)",
        default=0.85,
        min=0.1,
        max=2.5,
        step=5,
        precision=2
    )

    inertia: bpy.props.FloatProperty(
        name="모션 관성 반응",
        description="캐릭터 애니메이션 가속도에 가슴이 반응하는 관성 강도입니다",
        default=1.2,
        min=0.0,
        max=3.0,
        step=5,
        precision=2
    )

    idle_motion: bpy.props.FloatProperty(
        name="호흡 미동 강도",
        description="가만히 서 있을 때에도 미세하게 숨 쉬듯 움직이는 기본 호흡 미동 강도입니다 (0.0으로 설정 시 순수 이동 관성만 작동)",
        default=0.4,
        min=0.0,
        max=2.0,
        step=5,
        precision=2
    )

    is_preview_running: bpy.props.BoolProperty(
        name="프리뷰 실행 중",
        default=False,
        options={'SKIP_SAVE'}
    )

    # ── 4. 인게임 주입 및 디버그 ──
    target_game: bpy.props.EnumProperty(
        name="대상 게임",
        description="주입할 모드가 속한 게임입니다. 게임마다 스키닝 파이프라인 구조가 달라 주입 방식이 달라집니다",
        items=[
            ('ZZZ', "젠레스 존 제로 (ZZMI)", "vb0 교체 방식으로 주입합니다"),
            ('HSR', "붕괴: 스타레일 (SRMI)", "SRMI 스키닝 무결성 보존 + DrawIndexed 포스트-스키닝 관성 연산 방식으로 주입합니다"),
        ],
        default='ZZZ'
    )

    target_mod_dir: bpy.props.StringProperty(
        name="타겟 모드 폴더",
        description="바스트모핑 셰이더를 주입할 모드 폴더(또는 .ini 파일) 경로입니다",
        subtype='NONE'
    )

    merge_existing: bpy.props.BoolProperty(
        name="기존 물리 설정 유지",
        description="켜두면 이미 다른 파츠에 적용된 물리 데이터를 지우지 않고 가슴 물리를 함께 주입합니다",
        default=True
    )

    debug_mode: bpy.props.BoolProperty(
        name="디버그 모드 (인게임 토글)",
        description="켜두면 인게임에서 단축키로 바스트모핑 ON/OFF를 실시간 비교할 수 있습니다",
        default=False
    )

    debug_toggle_key: bpy.props.StringProperty(
        name="토글 단축키",
        description="인게임에서 바스트모핑을 켜고(ON) 끌(OFF) 단축키입니다 (예: h)",
        default="h"
    )


classes = (
    XXMI_PG_custom_physics_props,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.xxmi_custom_physics_props = bpy.props.PointerProperty(
        type=XXMI_PG_custom_physics_props
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "xxmi_custom_physics_props"):
        del bpy.types.Scene.xxmi_custom_physics_props
