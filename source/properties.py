import bpy


def on_bone_item_name_update(self, context):
    try:
        from .bone.data import handle_bone_item_rename
        handle_bone_item_rename(self, context)
    except Exception:
        pass


class XXMI_PG_bone_item(bpy.types.PropertyGroup):
    """개별 물리/모핑 본을 독립 슬롯으로 관리하며 고유한 가중치 및 파라미터를 보관합니다."""
    name: bpy.props.StringProperty(
        name="본 이름",
        default="새 모핑 본",
        description="모핑 본의 식별 이름입니다 (예: LeftChest, RightChest, Ponytail, Cape 등)",
        update=on_bone_item_name_update
    )
    prev_name: bpy.props.StringProperty(
        name="이전 본 이름",
        default=""
    )
    bone_type: bpy.props.EnumProperty(
        name="본 종류",
        items=[
            ('MORPH', "모핑 본", "3D 형태를 추종하여 부드럽게 변형되는 모핑 물리 본"),
        ],
        default='MORPH'
    )
    # JSON 인코딩 문자열: {"오브젝트이름": [정점인덱스, ...], ...}
    data_json: bpy.props.StringProperty(
        name="정점 데이터",
        default="{}",
        description="오브젝트별 등록된 정점 인덱스 목록(JSON)"
    )
    vert_count: bpy.props.IntProperty(
        name="정점 수",
        default=0,
        description="이 본 슬롯에 등록된 총 정점 개수입니다"
    )

    # ── 개별 본 가중치(웨이트) 설정 ──
    peak_weight: bpy.props.FloatProperty(
        name="최대 가중치 (끝단 강도)",
        description="이 본의 꼭짓점 최대 가중치입니다 (기본값: 0.40)",
        default=0.40,
        min=0.01,
        max=1.0,
        step=5,
        precision=2
    )
    falloff_power: bpy.props.FloatProperty(
        name="가중치 퍼짐 (Falloff)",
        description="가중치가 퍼지는 감쇠 곡선 지수입니다 (기본값: 1.0)",
        default=1.0,
        min=0.1,
        max=4.0,
        step=10,
        precision=2
    )
    smooth_iterations: bpy.props.IntProperty(
        name="경계 부드러움 (스무딩)",
        description="메쉬 표면을 따라 가중치를 부드럽게 블렌딩하는 횟수입니다 (기본값: 2)",
        default=2,
        min=0,
        max=8
    )
    lock_registered_verts: bpy.props.BoolProperty(
        name="등록된 영역만 유지",
        description="가중치를 다시 계산할 때 등록된 정점 영역만 유지합니다",
        default=True
    )

    # ── 개별 본 물리 거동 파라미터 ──
    strength: bpy.props.FloatProperty(
        name="흔들림 크기",
        description="이 본의 흔들리는 전체적인 움직임 폭 배율입니다 (기본값: 1.0)",
        default=1.0,
        min=0.0,
        max=3.0,
        step=5,
        precision=2
    )
    speed: bpy.props.FloatProperty(
        name="탄력 속도",
        description="흔들릴 때 튕겨오르는 속도감입니다 (기본값: 1.1)",
        default=1.1,
        min=0.2,
        max=3.0,
        step=5,
        precision=2
    )
    damping: bpy.props.FloatProperty(
        name="출렁임 멈춤 (감쇠)",
        description="흔들린 뒤 제자리로 멈추는 빠르기입니다 (기본값: 0.85)",
        default=0.85,
        min=0.1,
        max=2.5,
        step=5,
        precision=2
    )
    inertia: bpy.props.FloatProperty(
        name="이동 관성 반응",
        description="캐릭터 이동 시 관성에 반응하는 정도입니다 (기본값: 1.2)",
        default=1.2,
        min=0.0,
        max=3.0,
        step=5,
        precision=2
    )
    idle_motion: bpy.props.FloatProperty(
        name="숨쉬기 미동",
        description="정지 상태에서 숨 쉬듯 미세하게 움직이는 정도입니다 (기본값: 0.4)",
        default=0.4,
        min=0.0,
        max=2.0,
        step=5,
        precision=2
    )


def get_target_bone_enum_items(self, context):
    items = [
        ('AUTO', "자동 (최근접 모핑 본)", "3D 위치에 따라 가장 가까운 모핑 본을 자동으로 추종합니다"),
    ]
    seen_ids = {'AUTO'}

    # 1. 씬의 bone_items에 등록된 모핑 본들 동적 추가
    if context and hasattr(context, "scene") and hasattr(context.scene, "xxmi_custom_physics_props"):
        props = context.scene.xxmi_custom_physics_props
        for it in props.bone_items:
            if it.name and it.name not in seen_ids:
                seen_ids.add(it.name)
                items.append((it.name, f"{it.name} (모핑 본)", f"'{it.name}' 모핑 본에 100% 고정되어 따라 움직입니다"))

    # 2. 씬 내 가상 물리 스켈레톤(XXMI_Physics_Rig)에 실제 존재하는 본들 추가
    rig = None
    if context and hasattr(context, "scene") and hasattr(context.scene, "xxmi_custom_physics_props"):
        rig = context.scene.xxmi_custom_physics_props.anchor_obj
    if not rig:
        rig = bpy.data.objects.get("XXMI_Physics_Rig") or bpy.data.objects.get("XXMI_Breast_Rig")
    if rig and rig.type == 'ARMATURE':
        for b in rig.data.bones:
            if b.name not in seen_ids:
                seen_ids.add(b.name)
                items.append((b.name, f"{b.name} (모핑 본)", f"스켈레톤 내 '{b.name}' 본에 100% 고정됩니다"))

    if 'CENTER' not in seen_ids:
        items.append(('CENTER', "중앙 일체화 (모핑 본 중첩)", "양쪽 모핑 본의 움직임을 균일하게 공유하여 매듭/버클이 찢어지지 않습니다"))

    return items


class XXMI_PG_rigid_item(bpy.types.PropertyGroup):
    """개별 강체 부품(단추, 무전기, 리본 등)을 독립 슬롯으로 관리하는 데이터 구조입니다."""
    name: bpy.props.StringProperty(
        name="강체 이름",
        default="새 강체",
        description="강체 부품의 식별 이름입니다"
    )
    target_bone: bpy.props.StringProperty(
        name="목표 본",
        default="",
        description="이 강체가 100% 일체형으로 따라갈 대상 모핑 본입니다 (비워두면 위치 기반 자동 추종)"
    )
    # JSON 인코딩 문자열: {"오브젝트이름": [정점인덱스, ...], ...}
    data_json: bpy.props.StringProperty(
        name="정점 데이터",
        default="{}",
        description="오브젝트별 등록된 정점 인덱스 목록(JSON)"
    )
    vert_count: bpy.props.IntProperty(
        name="정점 수",
        default=0,
        description="이 강체 슬롯에 등록된 총 정점 개수입니다"
    )
    weight_scale: bpy.props.FloatProperty(
        name="가중치 배율",
        default=1.0,
        min=0.1,
        max=2.0,
        step=5,
        precision=2,
        description="부착된 모핑 표면 대비 강체의 흔들림 크기 배율입니다 (기본값: 1.0 = 표면과 100% 동일 밀착)"
    )


class XXMI_PG_custom_physics_props(bpy.types.PropertyGroup):
    # ── 1. 대상 메쉬 및 물리 본 ──
    target_obj: bpy.props.PointerProperty(
        name="대상 메쉬",
        type=bpy.types.Object,
        description="물리를 적용할 캐릭터 모드 메쉬 오브젝트입니다",
        poll=lambda self, obj: obj.type == 'MESH'
    )

    anchor_obj: bpy.props.PointerProperty(
        name="가상 물리 리그",
        type=bpy.types.Object,
        description="모핑 물리 움직임의 중심이 되는 본 오브젝트(XXMI_Physics_Rig)입니다",
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )

    # ── 본 종류 선택 ──
    bone_type: bpy.props.EnumProperty(
        name="본 종류",
        description="생성할 물리 본의 종류입니다",
        items=[
            ('MORPH', "모핑 본 (Morph Bone)", "자연스럽게 블렌딩되는 물리 모핑 본입니다"),
        ],
        default='MORPH'
    )

    # ── 다중 물리 본 목록 ──
    bone_items: bpy.props.CollectionProperty(
        type=XXMI_PG_bone_item,
        name="물리 본 목록"
    )
    bone_item_index: bpy.props.IntProperty(
        name="활성 본 인덱스",
        default=0
    )

    # ── 다중 강체 목록 ──
    rigid_items: bpy.props.CollectionProperty(
        type=XXMI_PG_rigid_item,
        name="강체 부품 목록"
    )
    rigid_item_index: bpy.props.IntProperty(
        name="활성 강체 인덱스",
        default=0
    )

    # ── 2. 가슴 가중치 감쇠 옵션 ──
    peak_weight: bpy.props.FloatProperty(
        name="최대 가중치 (끝단 강도)",
        description="가슴 꼭짓점의 최대 가중치입니다. 0.35~0.45(초록~파란색)으로 유지하면 노란색/빨간색 없이 부드럽고 유연하게 움직입니다 (기본값: 0.40)",
        default=0.40,
        min=0.1,
        max=1.0,
        step=5,
        precision=2
    )

    smooth_iterations: bpy.props.IntProperty(
        name="경계 부드러움 (스무딩)",
        description="옷이 가슴 전체를 덮고 있을 때 경계선이 각지거나 접히지 않도록 메쉬 표면을 따라 가중치를 부드럽게 블렌딩합니다 (0이면 스무딩 끔, 기본값: 2)",
        default=2,
        min=0,
        max=8,
    )

    falloff_power: bpy.props.FloatProperty(
        name="가중치 퍼짐 정도",
        description="가슴 가중치가 퍼지는 범위입니다. 기본값 1.0이며, 낮추면 전체로 부드럽게 퍼지고, 높이면 끝쪽으로 집중됩니다",
        default=1.0,
        min=0.3,
        max=3.0,
        step=10,
        precision=2
    )

    lock_registered_verts: bpy.props.BoolProperty(
        name="등록된 영역만 유지",
        description="가중치를 다시 계산할 때 처음에 등록한 정점 영역만 유지합니다",
        default=True
    )

    ui_show_advanced_breast: bpy.props.BoolProperty(
        name="고급 가중치 설정 펼치기",
        description="가중치 감쇠, 최대 가중치 등 세부 옵션을 펼치거나 접습니다",
        default=False
    )

    show_advanced_breast_settings: bpy.props.BoolProperty(
        name="세부 설정 및 관리 도구",
        description="가중치 세부 조절 및 초기화 도구를 펼치거나 접습니다",
        default=False
    )

    # ── 3. 가슴 물리 거동 파라미터 ──
    strength: bpy.props.FloatProperty(
        name="흔들림 크기",
        description="가슴이 흔들리는 전체적인 움직임 폭입니다 (0.0이면 정지, 1.0이 기본)",
        default=1.0,
        min=0.0,
        max=3.0,
        step=5,
        precision=2
    )

    speed: bpy.props.FloatProperty(
        name="탄력 속도",
        description="흔들릴 때 튕겨오르는 속도감입니다",
        default=1.1,
        min=0.2,
        max=3.0,
        step=5,
        precision=2
    )

    damping: bpy.props.FloatProperty(
        name="출렁임 멈춤 (감쇠)",
        description="흔들린 뒤 제자리로 멈추는 빠르기입니다 (낮을수록 오래 출렁이고, 높을수록 빨리 멈춤)",
        default=0.85,
        min=0.1,
        max=2.5,
        step=5,
        precision=2
    )

    inertia: bpy.props.FloatProperty(
        name="이동 관성 반응",
        description="달리거나 멈출 때 캐릭터 움직임에 가슴이 반응하는 정도입니다",
        default=1.2,
        min=0.0,
        max=3.0,
        step=5,
        precision=2
    )

    idle_motion: bpy.props.FloatProperty(
        name="숨쉬기 미동",
        description="가만히 서 있을 때 숨 쉬듯 미세하게 움직이는 정도입니다",
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
        description="물리를 적용할 모드의 대상 게임입니다",
        items=[
            ('ZZZ', "젠레스 존 제로 (ZZMI)", "젠레스 존 제로 모드 전용으로 최적화하여 주입합니다"),
            ('HSR', "붕괴: 스타레일 (SRMI)", "붕괴: 스타레일 모드 전용으로 최적화하여 주입합니다"),
        ],
        default='ZZZ'
    )

    target_mod_dir: bpy.props.StringProperty(
        name="모드 폴더",
        description="물리를 적용할 모드 폴더 경로입니다",
        subtype='NONE'
    )

    merge_existing: bpy.props.BoolProperty(
        name="기존 물리 설정 유지",
        description="이미 적용된 다른 물리 데이터가 있다면 지우지 않고 가슴 물리를 함께 주입합니다",
        default=True
    )

    debug_mode: bpy.props.BoolProperty(
        name="인게임 ON/OFF 토글 사용",
        description="게임 실행 중 단축키를 눌러 물리 적용 전/후를 실시간으로 비교할 수 있습니다",
        default=False
    )

    debug_toggle_key: bpy.props.StringProperty(
        name="토글 단축키",
        description="인게임에서 물리를 켜고 끌 키를 입력하세요 (기본: h)",
        default="h"
    )


classes = (
    XXMI_PG_bone_item,
    XXMI_PG_rigid_item,
    XXMI_PG_custom_physics_props,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.xxmi_custom_physics_props = bpy.props.PointerProperty(
        type=XXMI_PG_custom_physics_props
    )


def unregister():
    if hasattr(bpy.types.Scene, "xxmi_custom_physics_props"):
        del bpy.types.Scene.xxmi_custom_physics_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
