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

## 2. 프로젝트 구조 (Architecture) 및 세부 기술 문서
이 애드온은 독립된 단일 목적 블렌더 애드온으로 동작하며, 모든 소스코드는 `source/` 패키지 내부에 작업 흐름별 패널 단위로 모듈화되어 있습니다 (**`standalones/` 스크립트는 사용하지 않으며 오직 애드온으로만 동작합니다**).
- **공식 GitHub 저장소**: https://github.com/DPN-dpn/XXMI-Custom-Physics
- **세부 기술 문서 (상세 구현 및 수학적 알고리즘)**: [docs/README.md](docs/README.md)
  - [아키텍처 및 4단계 파이프라인](docs/ARCHITECTURE.md)
  - [해부학적 가중치 및 비침투 수학 알고리즘](docs/WEIGHTING_ALGORITHM.md)
  - [4-Substep ODE 및 인게임 HLSL 셰이더](docs/PHYSICS_AND_SHADERS.md)
  - [3DMigoto 버퍼 매칭 및 INI 주입 파이프라인](docs/INJECTOR_PIPELINE.md)

```text
XXMI-Custom-Physics/
├── __init__.py                 # 블렌더 애드온 메인 진입점 (bl_info 및 source 패키지 등록/해제)
├── docs/                       # [세부 기술 문서] 신규 에이전트 및 개발자를 위한 기술 설명서
│   ├── README.md               # 문서 메인 색인
│   ├── ARCHITECTURE.md         # 전체 아키텍처 및 모듈 연계
│   ├── WEIGHTING_ALGORITHM.md  # 가중치 계산, 비침투 배리어, 심 동기화
│   ├── PHYSICS_AND_SHADERS.md  # 물리 ODE 적분 및 인게임 HLSL 셰이더
│   └── INJECTOR_PIPELINE.md    # 3DMigoto 버퍼 매칭 및 INI 주입
└── source/
    ├── __init__.py             # 하위 모듈/패널 통합 레지스트리
    ├── utils.py                # 공통 상수, 좌표계/기하학 유틸리티, 공통 도움말 툴팁 오퍼레이터
    ├── properties.py           # UI 입력 및 상태 저장용 PropertyGroup (XXMI_PG_custom_physics_props)
    ├── bone/                   # [패널 1] 1. 물리 본 & 웨이트 설정 (bl_order = 0)
    │   ├── __init__.py
    │   ├── operators.py        # 본 종류별 생성, 해부학적 물리 본 배치, Envelope 기반 가중치 베이크 및 동기화
    │   └── panel.py            # XXMI_PHYS_PT_bone 패널 UI
    ├── rigid/                  # [패널 2] 2. 강체(액세서리) 설정 (bl_order = 1)
    │   ├── __init__.py
    │   ├── operators.py        # 다중 강체 슬롯 UIList 관리, 다중 메쉬 정점 바인딩 및 목표 본(Target Bone) 지정
    │   └── panel.py            # XXMI_PHYS_PT_rigid 패널 UI
    ├── simulation/             # [패널 3] 3. 물리 파라미터 & 프리뷰 (bl_order = 2)
    │   ├── __init__.py
    │   ├── operators.py        # 뷰포트 실시간 바운스/관성/호흡 물리 프리뷰 (Shape Key 및 프레임 핸들러)
    │   └── panel.py            # XXMI_PHYS_PT_simulation 패널 UI
    ├── injector/               # [패널 4] 4. 모드에 물리 주입 (bl_order = 3)
    │   ├── __init__.py
    │   ├── operators.py        # .ini 및 Position.buf KD-Tree 매칭, CustomPhysicsBones/Map.buf 및 2-Pass HLSL 생성, .ini 주입/롤백
    │   └── panel.py            # XXMI_PHYS_PT_injector 패널 UI
    └── updater/                # [패널 5] 업데이트 (bl_order = 9, DEFAULT_CLOSED)
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
3. **독립적인 물리 본 (Independent Physics Bones)**:
   - 좌/우 하드코딩이나 임의의 편향, 공유 루트 앵커 등의 꼼수를 전면 배제합니다.
   - 각 물리 본(`Breast.L`, `Breast.R` 등 모든 임의의 본)은 고유한 독립 루트 버텍스 앵커(`ROOT_VERTEX_IDX_L`, `ROOT_VERTEX_IDX_R`)와 독립적인 스프링/관성 시뮬레이션 상태(`PhysState`)를 가지며 대칭성과 독립성을 엄격히 보장합니다.
4. **C1 연속 해부학적 가중치 감쇠 (Hann Cosine & Calibrated Falloff)**:
   - 가중치는 전방 깊이 코사인 램프(흉벽 0.0 → 꼭짓점 1.0)와 방사형 Hann 코사인 감쇠(중심축 0.0 → 외곽 1.0)의 합성으로 산출합니다.
   - 기저 지수 보정(`0.35 * falloff_power`)을 적용하여, UI 기본값 1.0에서 끝에 몰리지 않고 풍성하고 자연스러운 볼륨 곡면을 형성하도록 합니다.
   - 가슴골/중앙 등 인접 본의 영향권이 중첩되는 영역은 양쪽 비율을 보존하면서 합이 `peak_weight`를 넘지 않도록 크로스 블렌딩 정규화합니다.
5. **다층 메쉬 절대 방사형 비침투 배리어 및 경계면 완만화 (Strict Radial Multi-Layer Non-Penetration & Boundary Taper)**:
   - **대원칙**: "본 중심에 가까운 안쪽 메쉬(속살)는 본 중심에서 더 먼 바깥쪽 메쉬(겉옷/장식)를 절대로 넘지 못한다 ($w_{\text{inner}} \le w_{\text{outer}}$)."
   - **방사형 레이어 매칭 (Radial Shell Separation)**: 
     - 오류를 유발하는 표면 법선(Normal) 계산을 배제하고, 본 앵커(`Head`)로부터의 방사형 거리($r = \|P - \text{Head}\|$) 및 방사선(Ray, $10^\circ$ 이내)을 기준으로 속살과 겉옷을 명확히 판별합니다.
     - 곡면 경사(아랫가슴 Underboob 등)에 구애받지 않고 본 중심에서 바깥으로 나가는 동일 볼륨 선상의 레이어를 $100\%$ 매칭합니다.
   - **Conformal Promotion & Ceiling Clamp**: 
     - 겉옷은 안쪽 살 이상으로 상향 일치($w_{\text{outer}} = \max(w_{\text{outer}}, w_{\text{inner}})$)하고,
     - 속살도 겉옷을 넘지 못하도록 상한선($w_{\text{inner}} = \min(w_{\text{inner}}, w_{\text{outer}})$)으로 엄격 클램프하여 속살 돌출을 수학적으로 원천 봉쇄합니다.
   - **경계면 완만화 (Boundary Gradient Taper)**: 
     - 모드 폴더 내의 모든 연관 `.ib` 인덱스 버퍼를 자동으로 수집하여 삼각망 인접 그래프를 구축합니다.
     - 단일 폴리곤 엣지 상에서 가중치가 $0.40 \to 0.00$으로 급락해 발생하는 인게임 메쉬 찢어짐(Tearing)을 방지하기 위해, 5회 패스에 걸쳐 연결된 엣지 가중치 편차를 $\Delta w \le 0.06$ 이내로 완만하게 감쇠합니다.
6. **UV 심(Seam) 및 하드 엣지 찢어짐 방지 (절대 최종 단계 동기화)**:
   - 블렌더에서 1개의 정점이 익스포트 시 UV 경계나 날카로운 모서리(Sharp Edge)에서 여러 개의 `.buf` 정점으로 분리됩니다.
   - 모든 스키닝 가중치 계산, 정규화, 스무딩, 비침투 처리가 끝난 **절대 최종 단계**에서 3D 공간 상의 동일 좌표(기본 $1.2\text{mm}$ 이내, DSU 그룹)를 공유하는 모든 분리 정점의 가중치를 100% 동일하게 일치시켜야 인게임에서 메쉬가 찢어지지 않습니다.
7. **물리 ODE 파라미터 완전 일치화 (Blender Preview ↔ In-Game HLSL)**:
   - 블렌더 뷰포트 프리뷰(`engine.py`)와 인게임 셰이더(`hsr.py`, `zzz.py`)의 물리 적분 공식(4-Substep Symplectic Euler)과 감쇠비(`damp = 2.0 * omega * (damping * 0.55)`), 관성비(`12.0 * inertia`), 최대 변위 클램프(`0.035 * max_strength`)를 100% 일치시켜 블렌더 프리뷰와 인게임 모션의 차이를 없앱니다.
   - 블렌더 프리뷰 시 몸통 회전 요(Yaw) 및 롤(Roll)에 의한 좌/우 차동 가속도를 부여하여 대칭 본들이 완전히 독립된 3D 물리 거동을 보이도록 합니다.
8. **강체 악세서리(Rigid Accessory) 보존**:
   - 가슴이나 옷 위에 있는 단추, 브로치, 리본 장식, 벨트 버클 등이 모핑으로 휘어지는 것을 막기 위해, `XXMI_Role_Rigid` 그룹에 속한 정점은 **연결된 메쉬 덩어리(Connected Mesh Island) 전체의 중심점(Centroid) 또는 표면 간접 추종 가중치로 통일**합니다.
9. **DirectX (DXGI) ↔ 블렌더 좌표계 변환 주의**:
   - 3DMigoto 버퍼(`Position.buf`)는 DirectX Y-Up 계열 좌표계를 사용하며, Blender 메쉬는 Z-Up 계열 좌표계를 사용합니다.
   - 버퍼 직접 파싱 및 위치 매칭 시 축 변환(`(px, -pz, py)`)을 반드시 일치시켜야 앵커 및 버텍스 위치 오차가 발생하지 않습니다.
10. **3DMigoto INI 주입 규칙**:
   - `vb0 = Resource...` 뿐만 아니라 최신 XXMI Tools의 `vb0 = ref Resource...`, `vb0 = copy Resource...` 형태를 모두 파싱·매칭해야 합니다.
   - 교체 시 `; [XXMI-PHYSICS-VB-START]`와 `; [XXMI-PHYSICS-ORIG] <원본 줄>` 주석 메타데이터를 함께 기록하여, 재주입이나 롤백 시 원본 `.ini` 구문을 100% 무손실 복원해야 합니다.
   - 디버그 모드(`debug_mode`) 활성화 시 `[KeyXXMIPhysToggle]` 섹션과 `if $xxmi_phys_enable == 1 ... else ... endif` 분기를 주입하여 인게임 단축키(기본 `VK_F9`)로 즉시 전/후 비교가 가능해야 합니다.
