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

    # ── 2. 장신구(강체) 감지 옵션 ──
    auto_detect_rigid: bpy.props.BoolProperty(
        name="장신구 자동 강체화",
        description="가슴 표면과 분리된 작은 독립 메쉬 덩어리(단추, 브로치, 펜던트 등)를 감지하여 젤리처럼 휘지 않고 단단하게 유지되도록 자동 처리합니다",
        default=True
    )

    rigid_max_verts: bpy.props.IntProperty(
        name="장신구 최대 버텍스 수",
        description="독립 메쉬 섬(Island) 중 정점 수가 이 값 이하인 작은 덩어리만 장신구로 분류합니다",
        default=120,
        min=3,
        max=1000
    )

    falloff_power: bpy.props.FloatProperty(
        name="가슴 감쇠 부드러움",
        description="가슴 꼭지점에서 몸통 둘레로 갈수록 가중치가 0으로 감쇠하는 부드러움 곡선 지수입니다 (높을수록 외곽이 단단히 고정됨)",
        default=1.6,
        min=0.5,
        max=4.0,
        step=10,
        precision=2
    )

    # ── 3. 가슴 물리 거동 파라미터 ──
    strength: bpy.props.FloatProperty(
        name="흔들림 강도 (진폭)",
        description="가슴 물리 바운스의 전체 진폭(크기) 배율입니다 (0.0이면 정지, 1.0이 기본)",
        default=1.0,
        min=0.0,
        max=3.0,
        step=5,
        precision=2
    )

    speed: bpy.props.FloatProperty(
        name="탄력 속도 (주파수)",
        description="가슴이 통통 튀는 복원 탄력 속도(고유 진동 주파수)입니다",
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
        name="이동/모션 관성 반응",
        description="캐릭터 이동, 점프, 대시, 공격 및 정지 시 몸통 가속도에 가슴이 반응하는 관성 강도입니다",
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
