# XXMI Custom Physics — 개발자 & 에이전트 종합 기술 문서

이 디렉터리는 **XXMI Custom Physics** 블렌더 애드온의 내부 아키텍처, 수학적 알고리즘, 물리 시뮬레이션, 인게임 HLSL 셰이더 주입 파이프라인을 다룹니다.  
새로운 세션의 AI 에이전트나 신규 개발자가 전체 코드베이스를 신속하게 파악하고 안전하게 유지보수·확장할 수 있도록 작성되었습니다.

---

## 📌 문서 색인 (Documentation Index)

| 문서 | 핵심 내용 |
|---|---|
| [1. 시스템 아키텍처 (ARCHITECTURE.md)](ARCHITECTURE.md) | 4단계 워크플로우, 모듈 구조, 프로퍼티 그룹, 버텍스 그룹 명명 규칙, UI 컨벤션 |
| [2. 가중치 알고리즘 (WEIGHTING_ALGORITHM.md)](WEIGHTING_ALGORITHM.md) | 해부학적 Hann Cosine 감쇠, 다층 방사형 비침투 배리어, 연결 섬 분리, UV 심 동기화 |
| [3. 물리 시뮬레이션 & 셰이더 (PHYSICS_AND_SHADERS.md)](PHYSICS_AND_SHADERS.md) | 4-Substep Symplectic Euler ODE, 블렌더-HLSL 상수 100% 일치화, 독립 루트 앵커, 좌표계 변환 |
| [4. 모드 주입 파이프라인 (INJECTOR_PIPELINE.md)](INJECTOR_PIPELINE.md) | 3DMigoto .ini 파싱, Position.buf KD-Tree 매칭 (0.0 정점 보존), Map.buf 생성, 롤백 및 토글 |

---

## 🎯 프로젝트 배경 및 핵심 철학

### 1. 해결하고자 하는 문제 (The Problem)
XXMI(GIMI, SRMI, ZZZMI, WWMI 등 3DMigoto 기반 게임 모딩) 시:
- 여캐 TS 모드의 **가슴(Breasts)**
- 단발 ➔ **장발(Long Hair)**, **포니테일/트윈테일**
- 바지 ➔ **스커트, 코트 자락, 망토**
- 오리지널 신설 파츠인 **꼬리(Tail), 날개(Wings)** 등

바닐라(원본) 캐릭터 스켈레톤에 해당 부위 물리 본이 없어, 모더들은 어쩔 수 없이 몸통(Spine)이나 머리(Head) 본에 뻣뻣하게 100% 웨이트를 칠하게 됩니다.  
기존의 정점 단위 단순 사인파 변위 셰이더는 **정점 간 거리가 왜곡되어 표면이 젤리처럼 꿀렁거리며(Jelly Wobble)** 옷 위의 단추나 금속 브로치까지 고무처럼 휘어지는 치명적 결함이 있었습니다.

### 2. 본 애드온의 해결책 (The Solution)
- **블렌더 내 가상 물리 스켈레톤(`XXMI_Physics_Rig`)**: 사용자가 N개의 임의 모핑 본 슬롯을 자유롭게 생성하고 본 위치/각도/반경을 직관적으로 조정.
- **해부학적 C1 연속 가중치 베이크**: 흉벽(0.0)에서 꼭짓점(1.0)으로 향하는 코사인 램프와 중심축 외곽 Hann Cosine 감쇠 곡선 합성.
- **3DMigoto Compute Shader (HLSL)**: 인게임에서 GPU 연산으로 다관절 FK 회전 행렬을 실시간 적분하고, 선형 블렌드 스키닝(LBS) 변위를 적용하여 **실제 뼈대가 심어진 것과 동일한 자연스러운 관절 회전 물리**를 구현.
- **강체(Rigid Accessory) 형태 보존**: 단추, 버클, 무전기, 리본 등은 연결된 메쉬 섬 단위로 하부 표면의 움직임을 간접 추종(Surface Proximity Transfer)하여 변형을 원천 차단.

---

## 📂 프로젝트 소스코드 맵

```text
XXMI-Custom-Physics/
├── __init__.py                 # 블렌더 애드온 진입점 (bl_info, register/unregister)
├── AGENTS.md                   # 에이전트 행동 가이드라인 및 프로젝트 원칙
├── README.md                   # 사용자 대상 릴리스 안내 문서
├── docs/                       # [본 폴더] 에이전트 & 개발자 전용 기술 문서
│   ├── README.md               # 문서 메인 색인
│   ├── ARCHITECTURE.md         # 전체 아키텍처 및 모듈 연계
│   ├── WEIGHTING_ALGORITHM.md  # 가중치 계산, 비침투 배리어, 심 동기화
│   ├── PHYSICS_AND_SHADERS.md  # 물리 ODE 적분 및 인게임 HLSL 셰이더
│   └── INJECTOR_PIPELINE.md    # 3DMigoto 버퍼 매칭 및 INI 주입
└── source/
    ├── __init__.py             # 서브패키지 통합 등록/해제
    ├── utils.py                # 공통 상수, 좌표계/기하학 연산, 도움말 오퍼레이터
    ├── properties.py           # UI 입력 및 상태 저장용 PropertyGroup (XXMI_PG_custom_physics_props)
    ├── bone/                   # [패널 1] 물리 본 슬롯 관리, 리그 생성/조정, 가중치 베이크
    │   ├── data.py             # 본 슬롯 정점 등록/해제/강체 차감 데이터 모델
    │   ├── operators.py        # 본 슬롯 UI 오퍼레이터 (추가, 삭제, 자동 리그 생성)
    │   ├── panel.py            # XXMI_PHYS_PT_bone 패널 UI (bl_order = 0)
    │   ├── rig.py              # XXMI_Physics_Rig 아마추어 생성 및 본 변환 추출
    │   ├── ui_list.py          # 본 슬롯 UIList
    │   └── weighting.py        # 해부학적 가중치 계산, 비침투 제약, 라플라시안 스무딩
    ├── rigid/                  # [패널 2] 강체 슬롯 관리
    │   ├── data.py             # 강체 슬롯 정점 등록/조회
    │   ├── operators.py        # 강체 추가, 삭제, 정점 등록 오퍼레이터
    │   ├── panel.py            # XXMI_PHYS_PT_rigid 패널 UI (bl_order = 1)
    │   └── ui_list.py          # 강체 슬롯 UIList
    ├── simulation/             # [패널 3] 물리 파라미터 및 뷰포트 프리뷰
    │   ├── engine.py           # 뷰포트 물리 엔진 (4-Substep Symplectic Euler)
    │   ├── operators.py        # 실시간 프리뷰 시작/정지, 가중치 재계산
    │   └── panel.py            # XXMI_PHYS_PT_simulation 패널 UI (bl_order = 2)
    ├── injector/               # [패널 4] 모드 파일 분석, 버퍼 매칭 및 INI 주입
    │   ├── buffer_io.py        # Position.buf, CustomPhysicsMap.buf 바이너리 I/O
    │   ├── hsr.py              # 붕괴: 스타레일(HSR) 2-Pass HLSL 생성 및 INI 주입
    │   ├── zzz.py              # 젠레스 존 제로(ZZZ) 2-Pass HLSL 생성 및 INI 주입
    │   ├── ini_parser.py       # 3DMigoto INI 섹션/리소스/키바인드 파서
    │   ├── loader.py           # 모드 폴더에서 Map.buf/INI 설정 역복원
    │   ├── operators.py        # 주입, 롤백, 설정 불러오기 오퍼레이터
    │   └── panel.py            # XXMI_PHYS_PT_injector 패널 UI (bl_order = 3)
    └── updater/                # [패널 5] 자동 업데이트 (bl_order = 9, 접힘 기본)
        ├── operators.py        # GitHub Releases 최신 버전 확인 및 자동 다운로드
        └── panel.py            # XXMI_PHYS_PT_updater 패널 UI
```

---

## ⚡ 에이전트 주의사항 및 퀵 팁 (Quick Rules for Agents)
1. **절대 블렌더 전용 API를 독립 스크립트로 취급하지 마세요**: `standalones/`는 영구 폐기되었으며 오직 Blender 애드온으로만 구동됩니다.
2. **블렌더 애드온 폴더 동기화**: 수정 사항이 발생하면 작업 폴더(`d:\VS Code\XXMI-Custom-Physics`)뿐만 아니라 블렌더 실행 폴더(`%APPDATA%\Blender Foundation\Blender\<버전>\scripts\addons\XXMI-Custom-Physics`)에도 즉시 파일을 복사/반영해야 사용자가 즉시 테스트할 수 있습니다.
3. **KDTree 검색 메서드**: `mathutils.kdtree.KDTree`는 `find_nearest()`가 아닌 `find(Vector)` 및 `find_range(Vector, radius)`를 사용합니다.
4. **버퍼 매칭 시 0.0 정점 누락 금지**: 4패널 모드 주입 시 `kd_unified`에 활성 정점뿐만 아니라 비활성(가중치 0.0) 몸통 정점까지 모두 포함해야 외곽으로 가중치가 번지지 않습니다.
