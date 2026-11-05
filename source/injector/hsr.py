import os
import re
import math
from .ini_parser import (
    _clean_injected_lines,
    _parse_ini_sections,
    _extract_res_name,
    backup_ini_file,
)
from ..bone.data import get_bone_params


def export_physics_motion_hlsl_hsr(export_dir, idx, root_vert_idx, props):
    """[단일 통합 Compute Shader 파이프라인으로 일원화되어 별도 모션 파일 미사용]"""
    pass


def export_vertex_physics_hlsl_hsr(export_dir, idx, vertex_count, props, peak_weight=1.0, root_vert_idx=0):
    active_bones = [it.name for it in getattr(props, "bone_items", []) if it.name.strip()]
    b0_name = active_bones[0] if len(active_bones) > 0 else "Breast.L"
    b1_name = active_bones[1] if len(active_bones) > 1 else "Breast.R"
    params_l = get_bone_params(props, b0_name)
    params_r = get_bone_params(props, b1_name)

    speed_l = max(0.2, float(params_l["speed"]))
    damping_l = max(0.1, float(params_l["damping"]))
    inertia_l = max(0.0, float(params_l["inertia"]))
    idle_l = max(0.0, float(params_l["idle_motion"]))
    strength_l = max(0.0, float(params_l["strength"]))

    speed_r = max(0.2, float(params_r["speed"]))
    damping_r = max(0.1, float(params_r["damping"]))
    inertia_r = max(0.0, float(params_r["inertia"]))
    idle_r = max(0.0, float(params_r["idle_motion"]))
    strength_r = max(0.0, float(params_r["strength"]))

    peak_w_l = float(params_l["peak_weight"])
    peak_w_r = float(params_r["peak_weight"])
    norm_l = (1.0 / peak_w_l) if peak_w_l > 0.05 else 1.0
    norm_r = (1.0 / peak_w_r) if peak_w_r > 0.05 else 1.0

    scale_l = norm_l * strength_l
    scale_r = norm_r * strength_r

    omega_l = 12.0 * speed_l
    stiff_k_l = omega_l * omega_l
    damp_c_l = 2.0 * omega_l * (damping_l * 0.55)

    omega_r = 12.0 * speed_r
    stiff_k_r = omega_r * omega_r
    damp_c_r = 2.0 * omega_r * (damping_r * 0.55)

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

static const float STIFF_K_L = {stiff_k_l:.4f};
static const float DAMP_C_L = {damp_c_l:.4f};
static const float INERTIA_SCALE_L = {inertia_l:.4f};
static const float IDLE_SCALE_L = {idle_l:.4f};
static const float SPEED_L = {speed_l:.4f};

static const float STIFF_K_R = {stiff_k_r:.4f};
static const float DAMP_C_R = {damp_c_r:.4f};
static const float INERTIA_SCALE_R = {inertia_r:.4f};
static const float IDLE_SCALE_R = {idle_r:.4f};
static const float SPEED_R = {speed_r:.4f};

static const float SCALE_L = {scale_l:.4f};
static const float SCALE_R = {scale_r:.4f};
static const float MAX_DISP = {0.045 * max(0.1, max(strength_l, strength_r)):.4f};

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

                // 실제 뼈대 애니메이션 이동 속도 기반 정직한 뉴턴 관성력 (F = -m*a)
                // 각 물리 본 슬롯은 독립된 관성력을 수신 (인위적인 좌우 왜곡/반전 제거)
                float3 raw_vel = delta / dt;
                float3 anim_vel = raw_vel / (1.0 + length(raw_vel) * 0.35);

                float3 ext_force_l = -anim_vel * (14.0 * INERTIA_SCALE_L);
                float3 ext_force_r = -anim_vel * (14.0 * INERTIA_SCALE_R);

                // 부드러운 자연 호흡 바운스 중첩
                if (IDLE_SCALE_L > 0.001) {{
                    float breath_t_l = time * (2.8 * SPEED_L);
                    ext_force_l += float3(0.0, sin(breath_t_l) * 1.5, cos(breath_t_l) * 1.0) * IDLE_SCALE_L;
                }}
                if (IDLE_SCALE_R > 0.001) {{
                    float breath_t_r = time * (2.8 * SPEED_R);
                    ext_force_r += float3(0.0, sin(breath_t_r) * 1.5, cos(breath_t_r) * 1.0) * IDLE_SCALE_R;
                }}

                float sub_dt = dt * 0.25;
                [unroll]
                for (int step = 0; step < 4; step++) {{
                    float3 accel_l = ext_force_l - STIFF_K_L * prev_disp_l - DAMP_C_L * spring_vel_l;
                    float3 accel_r = ext_force_r - STIFF_K_R * prev_disp_r - DAMP_C_R * spring_vel_r;

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

    if (total_w > 0.0001 && (SCALE_L > 0.0001 || SCALE_R > 0.0001)) {{
        float3 disp_l = s0.xyz * (bw.x * SCALE_L);
        float3 disp_r = s1.xyz * (bw.y * SCALE_R);

        // 좌우 본 영향이 겹치는 부위(가슴골 등)에서 변위가 무작정 2배로 폭발하지 않도록 정규화 선형 블렌딩
        float blend_norm = max(1.0, (bw.x + bw.y));
        float3 total_disp = (disp_l + disp_r) / blend_norm;

        float d_len = length(total_disp);
        if (d_len > MAX_DISP && d_len > 1e-5) {{
            total_disp = (total_disp / d_len) * MAX_DISP;
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
    backup_ini_file(ini_path)
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
