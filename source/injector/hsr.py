import os
import re
import math
from .common import (
    _clean_injected_lines,
    _parse_ini_sections,
    _extract_res_name,
)


def export_physics_motion_hlsl_hsr(export_dir, idx, root_vert_idx, props):
    """[단일 통합 Compute Shader 파이프라인으로 일원화되어 별도 모션 파일 미사용]"""
    pass


def export_vertex_physics_hlsl_hsr(export_dir, idx, vertex_count, props, peak_weight=1.0, root_vert_idx=0):
    """
    [4B단계: 실시간 뼈대 애니메이션 관성(Inertia) & 탄성 스프링 ODE 통합 셰이더]
    - 단일 패스 Compute Shader:
      1) 스레드 0: LiveMesh[ROOT_VERTEX_IDX]의 이동 델타(curr_root - prev_root)를 감지하여
         실제 뼈대 애니메이션 가속도에 반발하는 물리 관성력(Inertia force)과 탄성 스프링 ODE(4단계 서브스텝)를 계산,
         PhysState 버퍼에 기록.
      2) 전체 스레드: 가슴 정점에 계산된 물리 변위를 가중치 비율로 정밀 가산.
    - normalize(0) 배제로 NaN 방지, cs-t120 보존으로 타 모드 간섭 제로.
    """
    speed = max(0.2, float(getattr(props, "speed", 1.1)))
    damping = max(0.1, float(getattr(props, "damping", 0.85)))
    inertia = max(0.0, float(getattr(props, "inertia", 1.2)))
    idle_motion = max(0.0, float(getattr(props, "idle_motion", 0.6)))
    strength = max(0.0, float(getattr(props, "strength", 1.0)))
    norm_factor = (1.0 / peak_weight) if peak_weight > 0.05 else 1.0

    omega_0 = 12.0 * speed
    stiff_k = omega_0 * omega_0
    damp_c = 2.0 * omega_0 * (damping * 0.55)

    root_idx = root_vert_idx if (root_vert_idx is not None and root_vert_idx > 0) else (getattr(props, "root_vert_idx", 0) or 0)

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - Step 4B: Unified Animation Inertia & Spring Dynamics
// ============================================================================
struct Vertex {{
    float3 position;
    float3 normal;
    float4 tangent;
}};

Texture1D<float4> IniParams : register(t120);
StructuredBuffer<Vertex> LiveMesh : register(t50);
StructuredBuffer<Vertex> BaseMesh : register(t52);
StructuredBuffer<float4> BreastWeights : register(t53);
RWStructuredBuffer<Vertex> Output : register(u5);
RWStructuredBuffer<float4> PhysState : register(u6);

static const uint VERTEX_COUNT = {vertex_count};
static const uint ROOT_VERTEX_IDX = {root_idx};
static const float SPEED = {speed:.4f};
static const float INERTIA_SCALE = {inertia:.4f};
static const float IDLE_SCALE = {idle_motion:.4f};
static const float STIFF_K = {stiff_k:.4f};
static const float DAMP_C = {damp_c:.4f};
static const float NORM_FACTOR = {norm_factor:.4f};
static const float STRENGTH_SCALE = {strength:.4f};

[numthreads(64, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {{
    uint i = threadID.x;
    if (i >= VERTEX_COUNT) return;

    // 1. 현재 프레임에 적용할 물리 상태 읽기
    float4 s0 = PhysState[0];
    float4 s1 = PhysState[1];

    // 2. 스레드 0: LiveMesh를 추적하여 다음 프레임 물리 연산 수행
    if (threadID.x == 0) {{
        float time = IniParams.Load(int2(87, 0)).x;
        if (isnan(time) || isinf(time)) {{
            time = 0.0;
        }}

        float4 s2 = PhysState[2];
        float4 s3 = PhysState[3];
        float4 s4 = PhysState[4];

        float3 prev_disp_l  = s0.xyz;
        float3 prev_disp_r  = s1.xyz;
        float3 spring_vel_l = s2.xyz;
        float  prev_time    = s2.w;
        float3 spring_vel_r = s3.xyz;
        float  init_flag    = s3.w;
        float3 prev_root    = s4.xyz;

        float3 curr_root = LiveMesh[ROOT_VERTEX_IDX].position;
        if (isnan(curr_root.x) || isinf(curr_root.x) || length(curr_root) < 1e-4) {{
            curr_root = prev_root;
        }}

        if (init_flag < 0.5) {{
            prev_disp_l = float3(0, 0, 0);
            prev_disp_r = float3(0, 0, 0);
            spring_vel_l = float3(0, 0, 0);
            spring_vel_r = float3(0, 0, 0);
            prev_root = curr_root;
            prev_time = time;
            init_flag = 1.0;
        }} else {{
            float raw_dt = time - prev_time;
            if (abs(raw_dt) > 0.0005) {{
                float dt = clamp(raw_dt, 0.002, 0.04);
                float3 delta = curr_root - prev_root;

                // 텔레포트/컷씬 튐 방지
                if (length(delta) > 0.35 || raw_dt < 0.0 || raw_dt > 0.15) {{
                    delta = float3(0, 0, 0);
                    spring_vel_l = float3(0, 0, 0);
                    spring_vel_r = float3(0, 0, 0);
                }}

                // 실제 뼈대 애니메이션 이동 속도 기반 반작용 관성력
                float3 raw_vel = delta / dt;
                float3 anim_vel = raw_vel / (1.0 + length(raw_vel) * 0.35);
                float3 ext_force = -anim_vel * (14.0 * INERTIA_SCALE);

                // 부드러운 자연 호흡 바운스 중첩
                float breath_t = time * (2.8 * SPEED);
                float3 idle_f = float3(0.0, sin(breath_t) * 1.5, cos(breath_t) * 1.0) * IDLE_SCALE;
                ext_force += idle_f;

                float3 ext_force_l = ext_force;
                float3 ext_force_r = float3(-ext_force.x * 0.9, ext_force.y, ext_force.z);

                float sub_dt = dt * 0.25;
                [unroll]
                for (int step = 0; step < 4; step++) {{
                    float3 accel_l = ext_force_l - STIFF_K * prev_disp_l - DAMP_C * spring_vel_l;
                    float3 accel_r = ext_force_r - STIFF_K * prev_disp_r - DAMP_C * spring_vel_r;

                    spring_vel_l = clamp(spring_vel_l + accel_l * sub_dt, -3.0, 3.0);
                    spring_vel_r = clamp(spring_vel_r + accel_r * sub_dt, -3.0, 3.0);

                    prev_disp_l = clamp(prev_disp_l + spring_vel_l * sub_dt, -0.040, 0.040);
                    prev_disp_r = clamp(prev_disp_r + spring_vel_r * sub_dt, -0.040, 0.040);
                }}

                prev_root = curr_root;
                prev_time = time;
            }}
        }}

        PhysState[0] = float4(prev_disp_l, 0.0);
        PhysState[1] = float4(prev_disp_r, 0.0);
        PhysState[2] = float4(spring_vel_l, prev_time);
        PhysState[3] = float4(spring_vel_r, init_flag);
        PhysState[4] = float4(prev_root, 0.0);
    }}

    // 3. 각 정점별 가슴 변위 적용
    Vertex v = BaseMesh[i];
    float4 bw = BreastWeights[i];
    float total_w = bw.x + bw.y;

    if (total_w > 0.0001 && STRENGTH_SCALE > 0.0001) {{
        float3 disp_l = s0.xyz;
        float3 disp_r = s1.xyz;
        float3 total_disp = (disp_l * bw.x + disp_r * bw.y) * (NORM_FACTOR * STRENGTH_SCALE);

        float max_disp = 0.045 * max(0.1, STRENGTH_SCALE);
        float d_len = length(total_disp);
        if (d_len > max_disp && d_len > 1e-5) {{
            total_disp = (total_disp / d_len) * max_disp;
        }}

        v.position += total_disp;
    }}

    Output[i] = v;
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysics_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)


def inject_physics_hsr(ini_path, matched_buffers, props, master_idx):
    """
    [4B단계: 붕괴: 스타레일 실시간 뼈대 애니메이션 관성 & 스프링 통합 인젝터]
    """
    with open(ini_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    clean_lines, _ = _clean_injected_lines(raw_lines)
    toggle_key = getattr(props, "debug_toggle_key", "h").strip() or "h"

    breast_entries = [e for e in matched_buffers if e[4]]
    if not breast_entries:
        return

    master_entry = max(breast_entries, key=lambda e: e[3])
    master_idx = master_entry[0]
    master_buf_filename = master_entry[2]
    master_draw_sec = master_entry[8].strip().lower()

    # blend_sec -> entry 매핑 (대소문자 무시)
    blend_sec_map = {}
    for entry in breast_entries:
        blend_sec_name = entry[7].strip().lower()
        if blend_sec_name:
            blend_sec_map[blend_sec_name] = entry

    out_lines = []
    current_sec = None
    motion_injected = False

    for line in clean_lines:
        l_strip = line.strip()
        ll = l_strip.lower()

        if l_strip.startswith('[') and l_strip.endswith(']'):
            current_sec = l_strip[1:-1].strip()
            motion_injected = False
            out_lines.append(line)
            continue

        # 1. Blend 섹션 (DRAW_TYPE == 8 내부의 Resource\SRMI\PositionBuffer 교체)
        if current_sec and current_sec.lower() in blend_sec_map:
            entry = blend_sec_map[current_sec.lower()]
            idx = entry[0]

            if "resource\\srmi\\positionbuffer" in ll:
                parts = l_strip.split('=', 1)
                orig_rhs = parts[1].strip()
                indent = line[:len(line) - len(line.lstrip())]

                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-START]\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-ORIG] {l_strip}\n")
                out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                out_lines.append(f"{indent}    run = CustomShaderPhysics_{idx}\n")
                out_lines.append(f"{indent}    Resource\\SRMI\\PositionBuffer = ref ResourcePhysicsAnimated_{idx}\n")
                out_lines.append(f"{indent}else\n")
                out_lines.append(f"{indent}    {l_strip}\n")
                out_lines.append(f"{indent}endif\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-END]\n")
                continue

        # 2. DrawIndexed 섹션 (실시간 뼈대 애니메이션 관성 추적: LiveMesh = copy vb0)
        if current_sec and current_sec.lower() == master_draw_sec and not motion_injected:
            if ll.startswith("drawindexed") or ll.startswith("draw "):
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-START]\n")
                out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                out_lines.append(f"{indent}    ResourceLiveMesh_{master_idx} = copy vb0\n")
                out_lines.append(f"{indent}endif\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-END]\n")
                motion_injected = True

        out_lines.append(line)

    footer = "; [XXMI-PHYSICS-START]\n"
    if clean_lines and clean_lines[-1].strip():
        footer = "\n" + footer

    footer += f"""[Constants]
global persist $xxmi_phys_enable = 1

[KeyXXMIPhysToggle]
Key = {toggle_key}
type = cycle
$xxmi_phys_enable = 1, 0

[ResourceLiveMesh_{master_idx}]
type = StructuredBuffer
stride = 40
filename = {master_buf_filename}

[ResourcePhysicsState_{master_idx}]
type = RWStructuredBuffer
stride = 16
filename = CustomPhysics/CustomPhysicsState_{master_idx}.buf

"""

    for entry in breast_entries:
        idx, res_name, buf_filename, v_count, has_breast = entry[:5]
        dispatch_groups = math.ceil(v_count / 64.0)
        orig_cs_res = f"{res_name}CS"

        footer += f"""[CustomShaderPhysics_{idx}]
cs = CustomPhysics/CustomPhysics_{idx}.hlsl
x87 = time
cs-t120 = IniParams
cs-t50 = ref ResourceLiveMesh_{master_idx}
cs-t52 = ref {orig_cs_res}
cs-t53 = ref ResourcePhysicsMap_{idx}
cs-u5 = ResourcePhysicsAnimated_{idx}
cs-u6 = ResourcePhysicsState_{master_idx}
Dispatch = {dispatch_groups}, 1, 1
cs-u5 = null
cs-u6 = null
cs-t50 = null
cs-t52 = null
cs-t53 = null

[ResourcePhysicsAnimated_{idx}]
type = RWStructuredBuffer
stride = 40
filename = {buf_filename}

[ResourcePhysicsMap_{idx}]
type = StructuredBuffer
stride = 16
filename = CustomPhysics/CustomPhysicsMap_{idx}.buf

"""

    footer += "; [XXMI-PHYSICS-END]\n"
    out_lines.append(footer)

    with open(ini_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
