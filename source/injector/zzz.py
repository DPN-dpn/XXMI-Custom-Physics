import os
import re
import math
from .ini_parser import (
    _clean_injected_lines,
    _parse_ini_sections,
    _extract_res_name,
    _find_motion_insert_index,
    backup_ini_file,
)


def export_physics_motion_hlsl_zzz(export_dir, idx, root_vert_idx, props):
    speed = max(0.2, float(getattr(props, "speed", 1.1)))
    damping = max(0.1, float(getattr(props, "damping", 0.85)))
    inertia = max(0.0, float(getattr(props, "inertia", 1.2)))
    idle_motion = max(0.0, float(getattr(props, "idle_motion", 0.4)))

    omega_0 = 12.0 * speed
    stiff_k = omega_0 * omega_0
    damp_c = 2.0 * omega_0 * (damping * 0.55)

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - Torso Inertia Tracker (Feedback-Free ODE Physics)
// ============================================================================
struct Vertex {{
    float3 position;
    float3 normal;
    float4 tangent;
}};

StructuredBuffer<Vertex> LiveMesh : register(t50);
Texture1D<float4> IniParams : register(t120);

RWStructuredBuffer<float4> PhysState : register(u6);

static const uint ROOT_VERTEX_IDX = {root_vert_idx};

static const float STIFF_K = {stiff_k:.4f};
static const float DAMP_C = {damp_c:.4f};
static const float INERTIA_SCALE = {inertia:.4f};
static const float IDLE_SCALE = {idle_motion:.4f};

[numthreads(1, 1, 1)]
void main() {{
    float time = IniParams.Load(int2(87, 0)).x;

    float4 s0 = PhysState[0];
    float4 s1 = PhysState[1];
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

    if (length(curr_root) < 1e-4) {{
        float breath_t = time * 2.5;
        float3 fb_l = float3(sin(breath_t) * 0.005, sin(breath_t) * 0.010, cos(breath_t) * 0.007) * IDLE_SCALE;
        float3 fb_r = float3(-sin(breath_t) * 0.005, sin(breath_t) * 0.010, cos(breath_t) * 0.007) * IDLE_SCALE;
        PhysState[0] = float4(fb_l, 0.5);
        PhysState[1] = float4(fb_r, 0.5);
        return;
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
        if (abs(raw_dt) <= 0.0005) {{
            return;
        }}

        float dt = clamp(raw_dt, 0.002, 0.04);
        float3 delta = curr_root - prev_root;

        if (length(delta) > 0.25 || raw_dt < 0.0 || raw_dt > 0.15) {{
            delta = float3(0, 0, 0);
            spring_vel_l = float3(0, 0, 0);
            spring_vel_r = float3(0, 0, 0);
        }}

        float3 raw_vel = delta / dt;
        float3 anim_vel = raw_vel / (1.0 + length(raw_vel) * 0.5);
        float3 ext_force = -anim_vel * (12.0 * INERTIA_SCALE);

        if (IDLE_SCALE > 0.001) {{
            float breath_t = time * 2.5;
            float3 idle_f = float3(sin(breath_t) * 0.6, sin(breath_t) * 1.5, cos(breath_t) * 0.9) * IDLE_SCALE;
            ext_force += idle_f;
        }}

        // Pre-overhaul L/R symmetric opposite X-axis force
        float3 ext_force_l = ext_force;
        float3 ext_force_r = float3(-ext_force.x * 0.9, ext_force.y, ext_force.z);

        float sub_dt = dt * 0.25;
        [unroll]
        for (int step = 0; step < 4; step++) {{
            float3 accel_l = ext_force_l - STIFF_K * prev_disp_l - DAMP_C * spring_vel_l;
            float3 accel_r = ext_force_r - STIFF_K * prev_disp_r - DAMP_C * spring_vel_r;

            spring_vel_l = clamp(spring_vel_l + accel_l * sub_dt, -2.5, 2.5);
            spring_vel_r = clamp(spring_vel_r + accel_r * sub_dt, -2.5, 2.5);

            prev_disp_l = clamp(prev_disp_l + spring_vel_l * sub_dt, -0.025, 0.025);
            prev_disp_r = clamp(prev_disp_r + spring_vel_r * sub_dt, -0.025, 0.025);
        }}

        prev_root = curr_root;
        prev_time = time;
    }}

    PhysState[0] = float4(prev_disp_l, length(spring_vel_l));
    PhysState[1] = float4(prev_disp_r, length(spring_vel_r));
    PhysState[2] = float4(spring_vel_l, prev_time);
    PhysState[3] = float4(spring_vel_r, init_flag);
    PhysState[4] = float4(prev_root, 0);
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysicsMotion_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)




def export_vertex_physics_hlsl_zzz(export_dir, idx, vertex_count, props, peak_weight=1.0):
    strength = max(0.0, float(getattr(props, "strength", 1.0)))
    norm_factor = (1.0 / peak_weight) if peak_weight > 0.05 else 1.0

    hlsl_code = f"""// ============================================================================
// XXMI Custom Physics - Breast Skinning Compute Shader (Displacement Pass)
// ============================================================================
struct Vertex {{
    float3 position;
    float3 normal;
    float4 tangent;
}};

StructuredBuffer<Vertex> BaseMesh : register(t52);
StructuredBuffer<float4> BreastWeights : register(t53);
StructuredBuffer<float4> PhysState : register(t54);
RWStructuredBuffer<Vertex> Output : register(u5);

static const uint VERTEX_COUNT = {vertex_count};
static const float NORM_FACTOR = {norm_factor:.4f};
static const float STRENGTH_SCALE = {strength:.4f};

[numthreads(64, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID) {{
    uint i = threadID.x;
    if (i >= VERTEX_COUNT) return;

    Vertex v = BaseMesh[i];
    float4 bw = BreastWeights[i];
    float wl = bw.x;
    float wr = bw.y;
    float total_w = wl + wr;

    if (total_w > 0.0001 && STRENGTH_SCALE > 0.0001) {{
        float3 disp_l = PhysState[0].xyz;
        float3 disp_r = PhysState[1].xyz;

        float3 total_disp = (disp_l * wl + disp_r * wr) * (NORM_FACTOR * STRENGTH_SCALE);

        float max_disp = 0.030 * max(0.1, STRENGTH_SCALE);
        float d_len = length(total_disp);
        if (d_len > max_disp && d_len > 1e-5) {{
            total_disp = (total_disp / d_len) * max_disp;
        }}

        v.position += total_disp;

        if (bw.z < 0.5) {{
            v.normal = normalize(v.normal + total_disp * 2.5);
        }}
    }}

    Output[i] = v;
}}
"""
    out_path = os.path.join(export_dir, f"CustomPhysics_{idx}.hlsl")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(hlsl_code)




def inject_physics_zzz(ini_path, matched_buffers, props, master_idx=None):
    backup_ini_file(ini_path)
    with open(ini_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    clean_lines, _ = _clean_injected_lines(raw_lines)

    use_debug = bool(getattr(props, "debug_mode", False))
    toggle_key = getattr(props, "debug_toggle_key", "h").strip() or "h"

    breast_entries = [e for e in matched_buffers if e[4]]
    if not breast_entries:
        return

    master_entry = max(breast_entries, key=lambda e: e[3])
    master_idx = master_entry[0]
    master_buf_filename = master_entry[2]
    master_draw_sec = master_entry[8].lower()

    # blend lookup: res_name.lower() -> entry
    blend_res_map = {entry[1].lower(): entry for entry in breast_entries}

    out_lines = []
    current_sec = None
    motion_injected = False

    motion_insert_idx = _find_motion_insert_index(clean_lines, master_draw_sec)

    for line_no, line in enumerate(clean_lines):
        l_strip = line.strip()
        ll = l_strip.lower()

        if l_strip.startswith('[') and l_strip.endswith(']'):
            current_sec = l_strip[1:-1].strip()
            motion_injected = False
            out_lines.append(line)
            continue

        # Master Draw 섹션에서 토글 분기와 무관하게 항상 실행되는 지점에 단 하나의 통합 모션 추적 패스 주입
        if line_no == motion_insert_idx and not motion_injected:
            indent = line[:len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-START]\n")
            if use_debug:
                out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                out_lines.append(f"{indent}    ResourceLiveMesh_{master_idx} = copy vb0\n")
                out_lines.append(f"{indent}    run = CustomShaderPhysicsMotion_{master_idx}\n")
                out_lines.append(f"{indent}endif\n")
            else:
                out_lines.append(f"{indent}ResourceLiveMesh_{master_idx} = copy vb0\n")
                out_lines.append(f"{indent}run = CustomShaderPhysicsMotion_{master_idx}\n")
            out_lines.append(f"{indent}; [XXMI-PHYSICS-MOTION-END]\n")
            motion_injected = True

        # Blend 섹션의 vb0 교체
        if (ll.startswith("vb0 ") or ll.startswith("vb0=")) and "resourcephysicsanimated_" not in ll:
            res_val = _extract_res_name(line.split("=", 1)[1])
            if res_val.lower() in blend_res_map:
                entry = blend_res_map[res_val.lower()]
                idx = entry[0]
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-START]\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-ORIG] {l_strip}\n")
                if use_debug:
                    out_lines.append(f"{indent}if $xxmi_phys_enable == 1\n")
                    out_lines.append(f"{indent}    run = CustomShaderPhysics_{idx}\n")
                    out_lines.append(f"{indent}    vb0 = ResourcePhysicsAnimated_{idx}\n")
                    out_lines.append(f"{indent}else\n")
                    out_lines.append(f"{indent}    {l_strip}\n")
                    out_lines.append(f"{indent}endif\n")
                else:
                    out_lines.append(f"{indent}run = CustomShaderPhysics_{idx}\n")
                    out_lines.append(f"{indent}vb0 = ResourcePhysicsAnimated_{idx}\n")
                out_lines.append(f"{indent}; [XXMI-PHYSICS-VB-END]\n")
                continue

        out_lines.append(line)

    footer = "; [XXMI-PHYSICS-START]\n"
    if clean_lines and clean_lines[-1].strip():
        footer = "\n" + footer

    if use_debug:
        footer += f"""[Constants]
global persist $xxmi_phys_enable = 1

[KeyXXMIPhysToggle]
Key = {toggle_key}
type = cycle
$xxmi_phys_enable = 1, 0

"""

    # Master Motion Shader & Unified Physics State Buffer
    footer += f"""[CustomShaderPhysicsMotion_{master_idx}]
cs = CustomPhysics\\CustomPhysicsMotion_{master_idx}.hlsl
x87 = time
w85 = $xxmi_phys_enable
cs-t120 = IniParams
cs-t50 = ref ResourceLiveMesh_{master_idx}
cs-u6 = ResourcePhysicsState_{master_idx}
Dispatch = 1, 1, 1
cs-u6 = null
cs-t50 = null
cs-t120 = null

[ResourceLiveMesh_{master_idx}]
type = StructuredBuffer
stride = 40
filename = {master_buf_filename}

[ResourcePhysicsState_{master_idx}]
type = RWStructuredBuffer
stride = 16
filename = CustomPhysics\\CustomPhysicsState_{master_idx}.buf

"""

    # 각 메쉬별 변위 적용 쉐이더 (모든 메쉬가 Master PhysicsState를 함께 참조하여 100% 동기화)
    for entry in breast_entries:
        idx, res_name, buf_filename, v_count, has_breast = entry[:5]
        dispatch_groups = math.ceil(v_count / 64.0)
        footer += f"""[CustomShaderPhysics_{idx}]
cs = CustomPhysics\\CustomPhysics_{idx}.hlsl
w85 = $xxmi_phys_enable
cs-t52 = ref ResourceCustomPhysicsExtracted_{idx}
cs-t53 = ref ResourcePhysicsMap_{idx}
cs-t54 = ref ResourcePhysicsState_{master_idx}
cs-u5 = copy ResourceCustomPhysicsExtracted_{idx}
ResourcePhysicsAnimated_{idx} = ref cs-u5
Dispatch = {dispatch_groups}, 1, 1
cs-u5 = null
cs-t52 = null
cs-t53 = null
cs-t54 = null

[ResourcePhysicsAnimated_{idx}]

[ResourcePhysicsMap_{idx}]
type = StructuredBuffer
stride = 16
filename = CustomPhysics\\CustomPhysicsMap_{idx}.buf

[ResourceCustomPhysicsExtracted_{idx}]
type = StructuredBuffer
stride = 40
filename = {buf_filename}

"""

    footer += "; [XXMI-PHYSICS-END]\n"
    out_lines.append(footer)

    with open(ini_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)



