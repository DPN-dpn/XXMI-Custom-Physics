# XXMI Custom Physics 에이전트 가이드

이 문서는 **XXMI Custom Physics** 블렌더 애드온 개발 시 AI 에이전트가 준수해야 할 규칙, 프로젝트 배경, 핵심 기술 아키텍처를 정의합니다.

---

## 1. 프로젝트 목적 및 배경
- **배경**: XXMI(3DMigoto 기반 GIMI, SRMI, ZZZMI, WWMI 등) 캐릭터 모딩 시 바닐라 모델에 없던 파츠를 추가하거나 스타일을 크게 변경하는 경우가 많습니다.
  - 예: 남캐 ➔ 여캐 TS 모드의 **가슴(Breasts)**
  - 예: 단발 ➔ **장발(Long Hair)**, 또는 기존 헤어를 **포니테일(Ponytail) / 트윈테일(Twin-tails)**로 변경
  - 예: 바지 ➔ **스커트(Skirt) / 코트 / 망토**
  - 예: 바닐라에 전혀 없는 **꼬리(Tail)** 또는 **날개(Wings)** 파츠 신설
- **문제점**: 바닐라 스켈레톤에 해당 부위의 물리 본(Bone)이 없기 때문에 어쩔 수 없이 몸통/머리 본에 통째로 딱딱하게 웨이트를 칠하게 되어 인게임에서 파츠가 뻣뻣하게 굳어버립니다.
- **해결 방향 (핵심 설계 사상)**:
  - 정점(Vertex) 단위로 단순 사인파 변위를 주면 정점 간 거리가 왜곡되어 **표면이 젤리처럼 꿀렁거리는 현상(Jelly Wobble)**이 발생하고, 가슴이나 옷 위에 얹힌 단추·장신구까지 고무처럼 휘어집니다.
  - 따라서 본 애드온은 **블렌더 내에서 '가상 물리 스켈레톤(`XXMI_Physics_Rig`)'을 배치·조정**하고, **3DMigoto Compute Shader(HLSL) 내부에서 다관절 FK(Forward Kinematics) 회전 행렬과 선형 블렌드 스키닝(LBS)을 실시간 연산**하여 실제 물리 본이 들어간 것과 같은 자연스러운 관절 회전을 구현합니다.

---

## 2. 프로젝트 구조 (Architecture)
이 애드온은 독립된 단일 목적 블렌더 애드온으로 동작하며, 모든 소스코드는 `source/` 패키지 내부에 작업 흐름별 패널 단위로 모듈화되어 있습니다 (**`standalones/` 스크립트는 사용하지 않으며 오직 애드온으로만 동작합니다**).
- **공식 GitHub 저장소**: https://github.com/DPN-dpn/XXMI-Custom-Physics

```text
XXMI-Custom-Physics/
├── __init__.py                 # 블렌더 애드온 메인 진입점 (bl_info 및 source 패키지 등록/해제)
└── source/
    ├── __init__.py             # 하위 모듈/패널 통합 레지스트리
    ├── utils.py                # 공통 상수, 좌표계/기하학 유틸리티, 공통 도움말 툴팁 오퍼레이터
    ├── properties.py           # UI 입력 및 상태 저장용 PropertyGroup (XXMI_PG_custom_physics_props)
    ├── breast/                 # [패널 1] 1. 가슴 선택 & 앵커 생성 (bl_order = 0)
    │   ├── __init__.py
    │   ├── operators.py        # 에디트 모드 선택 버텍스 분할(L/R), 연결된 메쉬 섬(Island) 기반 장신구(강체) 자동 보호, 앵커 본 배치 및 방사형 가중치 베이크
    │   └── panel.py            # XXMI_PHYS_PT_breast 패널 UI
    ├── simulation/             # [패널 2] 2. 가슴 물리 파라미터 & 프리뷰 (bl_order = 1)
    │   ├── __init__.py
    │   ├── operators.py        # 뷰포트 실시간 바운스/관성/호흡 물리 프리뷰 (Shape Key 및 프레임 핸들러)
    │   └── panel.py            # XXMI_PHYS_PT_simulation 패널 UI
    ├── injector/               # [패널 3] 3. 모드에 바스트모핑 주입 (bl_order = 2)
    │   ├── __init__.py
    │   ├── operators.py        # .ini 및 Position.buf KD-Tree 매칭, CustomPhysicsBones/Map.buf 및 2-Pass HLSL 생성, .ini 주입/롤백
    │   └── panel.py            # XXMI_PHYS_PT_injector 패널 UI
    └── updater/                # [패널 4] 업데이트 (bl_order = 9, DEFAULT_CLOSED)
        ├── __init__.py
        ├── operators.py        # GitHub 최신 릴리스 확인, 자동 다운로드/설치, 저장소 열기 오퍼레이터
        └── panel.py            # XXMI_PHYS_PT_updater 패널 UI
```

---

## 3. UI (패널) 디자인 컨벤션
1. **헤더 레이아웃 (커스텀 아이콘 표시)**:
   - `bl_label = ""`로 패널의 기본 이름을 비워둡니다.
   - `draw_header(self, context)`를 오버라이드하여 `layout.label(text="패널 이름", icon='원하는_아이콘')` 형태로 제목과 아이콘을 직접 그립니다.
2. **우측 상단 툴팁(도움말) 버튼 배치**:
   - 패널 내용 최상단 우측에 `?` 아이콘(`object.xxmi_phys_help_tooltip`)으로 도움말 툴팁을 제공합니다.
3. **입력 필드 정렬 (`use_property_split`)**:
   - 라벨과 입력 필드를 깔끔하게 정렬하기 위해 항상 아래 옵션을 활성화합니다:
     ```python
     layout.use_property_split = True
     layout.use_property_decorate = False
     ```

---

## 4. 블렌더 & 3DMigoto 기술 컨벤션 및 주의사항
1. **Property Group**: `PointerProperty` 선언 시 반드시 `type=` 인자를 명시합니다.
2. **뷰포트 가시성(Visibility) 보장**: 모드 전환(`bpy.ops.object.mode_set`)이나 메쉬 조작 전 대상 오브젝트의 `hide_get()`, `hide_select`, `hide_viewport`를 해제해야 런타임 에러가 발생하지 않습니다.
3. **UV 심(Seam) 및 하드 엣지 찢어짐 방지**:
   - 블렌더에서 1개의 정점이 익스포트 시 UV 경계나 날카로운 모서리(Sharp Edge)에서 여러 개의 `.buf` 정점으로 분리됩니다.
   - 스키닝 가중치를 계산할 때 **동일한 3D 좌표를 공유하는 모든 정점은 반드시 100% 동일한 본 인덱스와 가중치**를 갖도록 동기화해야 인게임에서 메쉬가 찢어지지 않습니다.
4. **강체 악세서리(Rigid Accessory) 보존**:
   - 가슴이나 옷 위에 있는 단추, 브로치, 리본 장식, 벨트 버클 등이 모핑으로 휘어지는 것을 막기 위해, `XXMI_Role_Rigid` 그룹에 속한 정점은 **연결된 메쉬 덩어리(Connected Mesh Island) 전체의 중심점(Centroid)에서 평가한 단일 가상 본 가중치로 통일**합니다.
5. **3DMigoto INI 주입 규칙**:
   - `vb0 = Resource...` 뿐만 아니라 최신 XXMI Tools의 `vb0 = ref Resource...`, `vb0 = copy Resource...` 형태를 모두 파싱·매칭해야 합니다.
   - 교체 시 `; [XXMI-PHYSICS-VB-START]`와 `; [XXMI-PHYSICS-ORIG] <원본 줄>` 주석 메타데이터를 함께 기록하여, 재주입이나 롤백 시 원본 `.ini` 구문을 100% 무손실 복원해야 합니다.
   - 디버그 모드(`debug_mode`) 활성화 시 `[KeyXXMIPhysToggle]` 섹션과 `if $xxmi_phys_enable == 1 ... else ... endif` 분기를 주입하여 인게임 단축키(기본 `VK_F9`)로 즉시 전/후 비교가 가능해야 합니다.
